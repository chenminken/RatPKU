"""
整个推理的入口，整合所有已知方法(除了sleap)
## 分步x    
1. 差分帧计算（可多进程）
2. 差分帧分类（需要分片）
3. pv分类（可多gpu多进程）
"""
import argparse
from datetime import datetime
from decord import VideoReader
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import time
import numpy as np
import pandas as pd
import multiprocessing
import functools

import pickle
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from multiprocessing import Manager,Process
# from utils import clipSelected_once
# multiprocessing.set_start_method('forkserver', force=True)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

target_clas_num = 1
max_length = 60
displacement = 10
step = 60
windows_size = 60
area_min = 20
gray_thres = 60
inter = 10
intra = 10

def _make_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        The `argparse.ArgumentParser` that defines the CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-r',
                        '--recursive',
                        action='store_true',
                        help='recursive find video')
    parser.add_argument("--video_suffix", type=str,default="mkv")
    parser.add_argument("-i", "--input_folder", type=str, help="input folder")
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        action="append",
        help=(
            "Path to trained model directory (with training_config.json). "
            "Multiple models can be specified, each preceded by --model."
        ),
    )
    #model
    parser.add_argument(
        "--model",
        type=str,
        default="sleap",
        help="model path.",
    )

    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["none", "rich", "json"],
        default="rich",
        help=(
            "Verbosity of inference progress reporting. 'none' does not output "
            "anything during inference, 'rich' displays an updating progress bar, "
            "and 'json' outputs the progress as a JSON encoded response to the "
            "console."
        ),
    )
    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference only on CPU. If not specified, will use available GPU.",
    )
    device_group.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help=(
            "Run training on the i-th GPU on the system. If 'auto', run on the GPU with"
            " the highest percentage of available memory."
        ),
    )
    device_group.add_argument(
        "-p",
        "--process_num",
        type=int,
        default=4,
        help=(
            "Run training on the i-th GPU on the system. If 'auto', run on the GPU with"
            " the highest percentage of available memory."
        ),
    )
    parser.add_argument("output_folder", type=str,default="output_video")
    # Deprecated legacy args. These will still be parsed for backward compatibility but
    # are hidden from the CLI help.
    return parser

# 剪裁选中的视频片段，以帧为单位
def clipSelected_once(vid, dst, start, end):
    assert isinstance(vid, cv2.VideoCapture)
    assert end > start
    # 设置指定帧
    vid.set(cv2.CAP_PROP_POS_FRAMES,max(start-1, 1))
    num = end - start +1
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"XVID"),30, (width,height),1)

    for i in range(num):
        succes, frame = vid.read()
        if not succes:
            break
        out.write(frame)
    out.release()


def cal_video(args_input):
    src, dst_csv = args_input
    print("cal_video", src)
    if Path(dst_csv).exists():
        print("file exist, ignore compute it. ", src)
        return
    try:
        vid = cv2.VideoCapture(src)
        succes, prev = vid.read()
    except cv2.error as error:
        return

    if not succes:
        print('no succes')
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
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
        # kernel = np.ones((2, 2), np.uint8)
        # erode = cv2.erode(diff,kernel=kernel, iterations=2)
        # contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        erode_sum = 0
        erode_count = 0
        # for idx, i in enumerate(contours):
        #     area = cv2.contourArea(i)
        #     erode_sum+= area
        #     erode_count += 1

        diffsum = np.sum(diff)
        wrt.write("{}, {}, {}\n".format(erode_sum, diffsum, erode_count))
        prev_gray = gray
        count = count + 1
    vid.release()

def filter_points(filename):
    df = pd.read_csv(filename, names=['erode_sum','diffsum', 'erode_count'],header=None)
    diff = df[(df['erode_sum'] > area_min * 2)]
    return [diff.index.values.tolist(), df]

def filter(points, min_length, intra,diff_df):
    res = []
    batch = {}
    info = {"avg": 0, "min": 1000000, "max":0, "count":1}
    for i in points:
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

def get_windows_dataframe(df,start,end,step,windows_size,max_length):
    # end-start是否少于windows_size
    if end - start <= windows_size - 1 and start + windows_size - 1 <= max_length:
        yield df.iloc[start: start + windows_size]
    else:
        for i in range(start,min(df.shape[0],end),step):
            ret_df = df.iloc[i:(i+windows_size),:]
            yield ret_df


def make_dataset_of_framediff(file_name, segment,csvdir,windows_size=60, step=60, seg_size=0,vid=None):
    csv_list = dict()
    location_list = []
    filename_stem = file_name.split('.')[0]
    signal_list = []
    signal_index_list = []
    all_time_id = []
    count = 0
    for row in segment:
        frame_60 = []
        start = row['st']
        end = row['end']
        csv_file = csvdir / file_name
        if csv_file not in csv_list:
            csv_list[csv_file] = pd.read_csv(csv_file,names=['erodesum','diffsum20','erode_sum'])
        csv = csv_list[csv_file]

        max_length = csv.shape[0]
        # 叠加
        for segment_60 in get_windows_dataframe(csv,start, min(end, max_length), step=step,windows_size=windows_size,max_length=max_length):
            start_index = segment_60.index.values[0]
            end_index = segment_60.index.values[-1]
            if end_index - start_index != windows_size - 1:
                continue
            dst = csvdir  / (filename_stem + '-' + str(start_index) + '-' + str(end_index) + '.mkv')
            # if not dst.exists():
            #     clipSelected_once(vid, str(dst), start_index, end_index)
            frame_60.append((start_index,end_index))
            signal_list.append(segment_60.to_numpy())
            signal_index_list.extend([count] * len(segment))
            all_time_id.extend([tid for tid in range(len(segment_60))])
            count += 1
        location_list.append((file_name.split('.')[0],frame_60))
    if len(signal_list) == 0:
        return 0, 0
    signal_np: np.ndarray = np.concatenate(signal_list, axis=0)
    signal_indexes = np.array(signal_index_list)
    signal_np = np.concatenate((signal_np,np.expand_dims(signal_indexes, axis=1)),axis=1)
    X = np.concatenate((signal_np,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_df = pd.DataFrame(X, columns=['erodesum','diffsum20','erode_count','id','time'])
    X_df['id'] = X_df['id'].astype('int')
    return location_list, X_df

def extract_from_csv(csv_file: str):
    extraction_settings = ComprehensiveFCParameters()
    from utils import filter_points, filter
    pass_points, diff_df = filter_points(csv_file)
    test_list = []
    test_index_list = []
    segment = filter(pass_points, inter, intra, diff_df) 
    count = 0
    all_time_id = []
    info_segment = []
    for row in segment:
        seq: pd.core.series.Series = diff_df.iloc[row['st']:max(int(row['st'])+60,int(row['end'])),:]
        test_list.append(seq.to_numpy())
        test_index_list.extend([count] * len(seq))
        all_time_id.extend([tid for tid in range(len(seq))])
        count += 1
        info_segment.append(f"{row['st']}-{row['end']}")

    # 拼装pandas dataframe格式
    if len(test_list) == 0:
        return 0, 0
    test_np: np.ndarray = np.concatenate(test_list,axis=0)
    test_indexes = np.array(test_index_list)
    test_np = np.concatenate((test_np,np.expand_dims(test_indexes, axis=1)),axis=1)
    X_test = np.concatenate((test_np,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_test = pd.DataFrame(X_test, columns=['erodesum','diffsum20','erode_count','diffsum10','id','time'])
    X_test['id'] = X_test['id'].astype('int')
    X_test = extract_features(X_test[['id','time','diffsum20','erodesum']], column_id='id', column_sort='time',
                        default_fc_parameters=extraction_settings,
                        # we impute = remove all NaN features automatically
                        impute_function=impute)
    
    return X_test, info_segment

def extract_feature_from_csv_and_segment(segment_list: list,csv_file: str,windows_size=60, step=60,src=None):
    """
    不切片
    Args:
    - segment, from filter_point
    - csv_file, str or path of cal_video output
    Outputs:
    - location_list
    """
    count = 0
    all_time_id = []
    location_list = [] #输出索引
    subset_list = [] # 用于暂时保存signal
    subset_index_list = [] #用于指定id
    subset_filename = [] # 用于定位视频片段
    csv_file = Path(csv_file)
    csv_df = pd.read_csv(csv_file, names=['erode_sum','diffsum', 'erode_count'],header=None)
    for row in segment_list:
        # 从文件名中寻找信息匹配
        filename_stem = csv_file.stem
        start = row['st']
        end = row['end']
        start = int(start)
        end = int(end)
        max_length = csv_df.shape[0]
        segment_str = f"{start}-{end}"
        for segment in get_windows_dataframe(csv_df,start, min(end, max_length), step=step,windows_size=windows_size,max_length=max_length):
            start_index = segment.index.values[0]
            end_index = segment.index.values[-1]
            if end_index - start_index != windows_size - 1:
                continue
            dst = csv_file.parent  / (filename_stem + '-' + str(start_index) + '-' + str(end_index) + '.mkv')
            # if not dst.exists():
            #     print("clip")
            #     clipSelected_once(vid, str(dst), start_index, end_index)
            location_tmp = {"start": start_index, "end": end_index, "filename": src,'id':count,'segment':segment_str, "csv_file":csv_file}
            location_list.append(location_tmp)
            subset_list.append(segment.to_numpy())
            subset_index_list.extend([count] * len(segment))
            all_time_id.extend([tid for tid in range(len(segment))])
            count += 1
    if len(subset_list) == 0:
        return 0, 0
    subset_np: np.ndarray = np.concatenate(subset_list,axis=0)
    subset_indexes = np.array(subset_index_list)
    X = np.concatenate((subset_np,np.expand_dims(subset_indexes, axis=1)),axis=1)
    # 拼装pandas dataframe格式
    X = np.concatenate((X,np.expand_dims(np.array(all_time_id),axis=1)),axis=1)
    X_df = pd.DataFrame(X, columns=['erodesum','diffsum20','erode_count','id','time'])
    X_df['id'] = X_df['id'].astype('int')
    location_pd = pd.DataFrame(location_list)
    return X_df, location_pd

def predict_signal(X_df, location_pd,clf):
    extraction_settings = ComprehensiveFCParameters()
    # # print(location_pd)
    # # print(X_df)
    feature = extract_features(X_df[['id','time','diffsum20']], column_id='id', column_sort='time',
                        default_fc_parameters=extraction_settings,
                        impute_function=impute,n_jobs=0)
    label_pr = clf.predict(feature)
    location_pd['predict'] = label_pr
    # location_pd['predict'] = 0
    return location_pd

def step2(args_input):
    """一个单独的实例,提取出差分帧信号并分类,针对单个cal_video csv文件"""
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)
    vid = cv2.VideoCapture(src)
    pass_points, diff_df = filter_points(dst_csv)
    segment = filter(pass_points, inter, intra, diff_df) 
    X_df, location = extract_feature_from_csv_and_segment(segment, dst_csv, windows_size=60, step=60,src=src)
    # print(X_df, location)
    vid.release()
    # 信号分类算法
    return [X_df, location]

def step3(location,clas,args_input,video_filename):
    if location is None:
        return
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)
    if (dst_csv.parent / (dst_csv.stem + "_res.csv")).exists():
        return
    vid = cv2.VideoCapture(str(video_filename))
    tmp_video_list = []
    for i in location.index:
        print(i)
        filename = Path(location.loc[i,'filename'])
        
        start = location.loc[i,'start']
        end = location.loc[i,'end']
        count = 0
        judge_list = []
        judge_score_list = []

        dst = dst_csv.parent  / (filename.stem + '-' + str(start) + '-' + str(end) + '.mkv')
        print(filename, dst)
        if not dst.exists():
            clipSelected_once(vid,str(dst),start,end)
        tmp_video_list.append(str(dst))
    results = clas.predict_batch(tmp_video_list)
    for result,i in zip(results,location.index):
        idmax =  int(result['class_ids'][0])
        score =  result['scores'][0]
        location.loc[i,'score'] = score
        location.loc[i, 'behavor'] = idmax # 0 for normal, 1 for scratching
        if idmax == target_clas_num:  # 如果是想要的抓挠类
            count += 1
    location.to_csv(dst_csv.parent / (dst_csv.stem + "_res.csv"),mode = 'w')
    vid.release()

def clip_predict(args_input):
    src, dst_csv = args_input
    dst_csv = Path(dst_csv)

def cpu_process(queue, i):
    """
    cpu工作，包含calvideo和后面的"""
    print("begin cpu_process", queue, i)
    with open('20221201-v4-ds20221118.pickle', 'rb') as fw:
        clf = pickle.load(fw)
    cal_video(i)
    src, dst_csv = i
    dst_csv = Path(dst_csv)
    src = Path(src)
    pass_points, diff_df = filter_points(dst_csv)
    segment = filter(pass_points, inter, intra, diff_df) 
    X_df, location = extract_feature_from_csv_and_segment(segment, dst_csv, windows_size=60, step=60, src=src)
    if (dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv")).exists():
        location_pd = pd.read_csv(str((dst_csv.parent/(dst_csv.stem+"_location_sig_sc.csv"))))
        video_filename = ''
    else:
        if isinstance(X_df, int):
            return
        location_pd = predict_signal(X_df, location,clf)
        if location_pd.shape[0] == 0:
            return
        video_filename = Path(location_pd.loc[0,'filename'])
        if (dst_csv.parent / (dst_csv.stem + "_res.csv")).exists():
            return
        vid = cv2.VideoCapture(str(video_filename))
        for j in location.index:
            print(j)
            filename = Path(location.loc[j,'filename'])
            
            start = location.loc[j,'start']
            end = location.loc[j,'end']


            dst = dst_csv.parent  / (filename.stem + '-' + str(start) + '-' + str(end) + '.mkv')
            print(filename, dst)
            if not dst.exists():
                clipSelected_once(vid,str(dst),start,end)
        location_pd.to_csv(video_filename.parent/(video_filename.stem+"_location_sig_sc.csv"))
    queue.put([i, video_filename])
    print('cpu part log: ', i ,video_filename, queue.empty())

class Func(object):
    def __init__(self):
        # 利用匿名函数模拟一个不可序列化象
        # 更常见的错误写法是，在这里初始化一个数据库的长链接
        self.num = lambda: None

    def work(self, num=None):
        self.num = num
        return self.num

    @staticmethod
    def call_back(res):
        print(f'Hello,World! {res}')

    @staticmethod
    def err_call_back(err):
        print(f'出错啦~ error：{str(err)}')

def main_split(args: list = None):
    """Entrypoint for `MOUSEACTION` CLI for running inference.

    Args:
        args: A list of arguments to be passed into mouse-action.
    """
    start_timestamp = str(datetime.now())
    logger.info("Started inference at:", start_timestamp)
    parser = _make_cli_parser()
    args, _ = parser.parse_known_args(args=args)
    logger.info("Args:")
    logger.info(vars(args))
    input_path = Path(args.input_folder)
    if input_path.is_file():
        video_list = [input_path]
    else:
        video_list = input_path.rglob(f"*.{args.video_suffix}") \
            if args.recursive \
            else input_path.glob(f"*.{args.video_suffix}")
    # 组装多进程的任务入参
    list_args = []
    video_list = [i for i in video_list]
    for video_filename in video_list:
        workdir =  video_filename.parent
        workdir = Path(str(workdir).replace(str(input_path),args.output_folder))
        workdir.mkdir(exist_ok=True,parents=True)
        newcsv_file = workdir / (video_filename.stem + ".csv")
        list_args.append([str(video_filename),str(newcsv_file)])
    
    queue = Manager().Queue()

    cpu_pool = multiprocessing.Pool(processes=4)
    for i in list_args:
        cpu_pool.apply_async(cpu_process,args=(queue,i),error_callback=Func.err_call_back)
    from ppvideo import PaddleVideo
    # TODO: 修改为入参
    model_str = args.model
    # model_str = "/home/chenmin/PaddleVideo-old/inference/ppTSM20230224rat-v2/ppTSM_rat_pku_20230224"
    # model_str = "/home/chenmin/PaddleVideo-old/inference/ppTSM20221208/ppTSM_mouse_20220613"
    clas = PaddleVideo(model_file= model_str +'.pdmodel',
                    params_file= model_str  + '.pdiparams',
                    use_gpu=True,use_tensorrt=False,batch_size=4)
    count = 0
    dest_count = len(list_args)
    while True:
        if count == dest_count:
            break
        if not queue.empty():
            print("init inference")
            arg_item, video_filename = queue.get(True)
            count += 1
            if video_filename == '':
                continue
            if (video_filename.parent/(video_filename.stem+"_location_sig_sc.csv")).exists():
                location = pd.read_csv(str((video_filename.parent/(video_filename.stem+"_location_sig_sc.csv"))))
                # print(arg_item, video_filename, location.loc[0,'filename'])
                scratching_location = location
                # scratching_location = location[location['predict'] == target_clas_num]
                step3(scratching_location, clas,arg_item, video_filename)
            else:
                print("video _location_sc don't exist")
        else:
            time.sleep(5)
    cpu_pool.close()
    cpu_pool.join()


def main(args: list = None):
    """Entrypoint for `MOUSEACTION` CLI for running inference.

    Args:
        args: A list of arguments to be passed into mouse-action.
    """
    start_timestamp = str(datetime.now())
    logger.info("Started inference at:", start_timestamp)
    parser = _make_cli_parser()
    args, _ = parser.parse_known_args(args=args)
    logger.info("Args:")
    logger.info(vars(args))
    input_path = Path(args.input_folder)
    if input_path.is_file():
        video_list = [input_path]
    else:
        video_list = input_path.rglob(f"*.{args.video_suffix}") \
            if args.recursive \
            else input_path.glob(f"*.{args.video_suffix}")
    # 组装多进程的任务入参
    list_args = []
    video_list = [i for i in video_list]
    for video_filename in video_list:
        workdir =  video_filename.parent
        workdir = Path(str(workdir).replace(str(input_path),args.output_folder))
        workdir.mkdir(exist_ok=True,parents=True)
        newcsv_file = workdir / (video_filename.stem + ".csv")
        list_args.append([str(video_filename),str(newcsv_file)])
    # 多进程执行第一步，cal_video
    p = multiprocessing.Pool(processes=10)
    with tqdm(total=len(list_args),) as t:
        for _ in p.imap(cal_video, list_args):
            t.update()
        p.close()
        p.join()

    from ppvideo import PaddleVideo
    
    # model_str = "/home/chenmin/PaddleVideo-old/inference/ppTSM20221208/ppTSM_mouse_20220613"
    model_str = args.model
    clas = PaddleVideo(model_file= model_str +'.pdmodel',
                    params_file= model_str  + '.pdiparams',
                    use_gpu=True,use_tensorrt=False,batch_size=16)
    for arg_item,video_filename in zip(list_args,video_list):
        if (video_filename.parent/(video_filename.stem+"_res.csv")).exists():
            location = pd.read_csv(str((video_filename.parent/(video_filename.stem+"_res.csv"))))
            print(arg_item, video_filename, location.loc[0,'filename'])
            scratching_location = location[location['predict'] == target_clas_num]
            step3(scratching_location, clas,arg_item, video_filename)

if __name__ == "__main__":
    main_split()
    # main()
    print("\n")