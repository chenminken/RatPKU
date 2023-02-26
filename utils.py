# 加载opencv
import cv2
import numpy as np
import time
import pandas as pd
import h5py
import csv
# from paddleclas import PaddleClas
from pathlib import Path
import pathlib
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

area_min = 20
area_max = 200
gray_thres = 60
def filter(points, min_length, intra,diff_df):
    res = []
    batch = {}
    info = {"avg": 0, "min": 1000000, "max":0, "count":1}
    for i in points:
        print(i)
        diff = diff_df.iloc[i,0]
        if 'end' in batch.keys() and i - batch['end'] < intra:
            batch['end'] = i
            info['avg'] = (info['avg'] + diff) / info['count']
            info['min'] = min(info['min'],diff)
            info['max'] = max(info['max'],diff)
        elif 'end' in batch.keys() and i - batch['end'] >= intra:
            if batch['end'] - batch['st'] >= min_length:
                info['avg'] = (info['avg'] + diff) / info['count']
                info['min'] = min(info['min'], diff)
                info['max'] = max(info['max'], diff)
                batch['info'] = info.copy()
                res.append(batch)
            batch = {}
            batch['st'] = i
            batch['end'] = i
            info = {"avg": 0, "min": 1000000, "max": 0, "count": 1}
        else:
            batch['st'] = i
            batch['end'] = i
            info = {"avg": diff, "min": diff, "max": diff, "count": 1}
        info['count'] += 1
    return res




def cal_video(src,dst_csv):
    print('cal_video')

    vid = cv2.VideoCapture(src)
    succes, prev = vid.read()

    if not succes:
        print('no succes')
        exit(-1)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    if pathlib.Path(dst_csv).exists():
        print("file exist, ignore it. ", src)
        return 0
    wrt = open(dst_csv, 'w+')
    count = 0
    launch = time.time()
    # vid.set(cv2.CAP_PROP_POS_MSEC, 60000)
    while vid.isOpened():
        succes, frame = vid.read()
        if not succes:
            break
        # 差分帧和颜色滤波合并
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #对原始图像作差
        diff = cv2.absdiff(gray, prev_gray)
        _, diff = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        erode = cv2.erode(diff,kernel=kernel, iterations=2)
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        erode_sum = 0
        erode_count = 0
        for idx, i in enumerate(contours):
            area = cv2.contourArea(i)
            erode_sum+= area
            erode_count += 1

        diffsum = np.sum(diff)
        wrt.write("{}, {}, {}\n".format(erode_sum, diffsum, erode_count))
        prev_gray = gray
        count = count + 1
        # print(count)
        # if count == 5000:
        #     break
    print(time.time()-launch)
    vid.release()
# 计算差分帧超过20000的点
def filter_points(filename):
    df = pd.read_csv(filename, names=['erode_sum','diffsum', 'erode_count'],header=None)
    diff = df[(df['erode_sum'] > area_min * 2)]
    return [diff.index.values.tolist(), df]

def cal_min_rect_w_h(rect):
    box = np.int0(cv2.boxPoints(rect))
    w = np.sqrt(np.sum(np.square(box[0] - box[1])))
    h = np.sqrt(np.sum(np.square(box[2] - box[1])))
    max_length = max(w,h)
    min_length = min(w,h)
    return min_length, max_length, box
# 剪裁选中的视频片段，以帧为单位
def clipSelected_once(vid, dst, start, end,is_feature=False):
    assert isinstance(vid, cv2.VideoCapture)
    assert end > start
    # 设置指定帧
    vid.set(cv2.CAP_PROP_POS_FRAMES,max(start-1, 1))
    num = end - start +1
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"XVID"),30, (width,height),1)
    if is_feature:
        succes, prev = vid.read()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        diff_list = []

    wrapper = np.zeros((height,width),np.uint8)
    erode_count = 0
    erode_num = 0
    erode_filtered = 0
    es_pred_list = [0,0,0]

    for i in range(num):
        succes, frame = vid.read()
        if not succes:
            break
        if is_feature:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, threshold = cv2.threshold(gray, gray_thres, 1, cv2.THRESH_BINARY_INV)
            _, pre_threshold = cv2.threshold(prev_gray, gray_thres, 1, cv2.THRESH_BINARY_INV)
            diff_origin = cv2.absdiff(gray, prev_gray)
            _, diff_threshold = cv2.threshold(diff_origin, 20, 1, cv2.THRESH_BINARY)
            diff_list.append(np.sum(diff_threshold))

            wrapper += diff_threshold
            prev_gray = gray
            out.write(frame)

        else:
            out.write(frame)
    out.release()
    # cv2.imshow("wrapper" + dst,wrapper)
    pref = dst.split('.')[:-1]
    postf = dst.split('.')[-1]
    maxid = es_pred_list.index(max(es_pred_list))

def clipSelected(src, dst_prefix, points):
    vid = cv2.VideoCapture(src,0)
    for idx, i in enumerate(points):
        dst = dst_prefix.format(idx, i['st'], i['end'])
        clipSelected_once(vid, dst, i['st'], i['end'])
        # print("a")
    vid.release()


def export_range(points,dst_prefix, df):
    for idx, i in enumerate(points):
        dst = dst_prefix.format(idx) + ".csv"
        selected_diff = df[(df.index >= i['st']) & (df.index <= i['end'])]
        selected_diff.to_csv(dst)



def export_csv(filename: str,output_filename: str):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        point_scores = f['point_scores'][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print("===tracking_scores shape===")
    print(point_scores.shape)

    cls = ['frames'] 
    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
        cls.append(name + '_x')
        cls.append(name + '_y')
        cls.append(name + '_score')

    print()

    res = []
    for i in range(locations.shape[0]):
        temp = []
        temp.append(i)
        for j in range(locations.shape[1]):
            if str(locations[i][j][0][0]) == 'nan':
                temp.append(0)
                temp.append(0)
                temp.append(0)
            else:
                temp.append(int(locations[i][j][0][0]))
                temp.append(int(locations[i][j][1][0]))
                temp.append(point_scores[i][j][0])
        res.append(temp)
        
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([item for item in cls])
        for m in res:
            writer.writerow(m)
            i+=1


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y
