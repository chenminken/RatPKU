"""
制作大鼠抓挠行为识别数据集
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re


def make_dataset(root_path: Path, padding_left=0, paddle_right=0):
    non_list = []
    sc_list = []
    non_index_list = []
    sc_index_list = []
    csv_list = dict()
    filenames_non = []
    filenames_sc = []
    count = 0
    y = []

    all_time_id = []
    for row in (root_path/"non").glob("*.mkv"):
        # 从文件名中寻找信息匹配
        filenames_non.append(row.name)
        filename_stem,_,start,end = re.search("(EZVZ\d+)-(\d+)-(\d+)-(\d+)",row.stem).groups()
        csv_file = root_path / "csv" / (filename_stem+".csv")
        if csv_file not in csv_list:
            csv_list[csv_file] = pd.read_csv(csv_file,names=['erodesum','diffsum20','erode_sum'])
        csv = csv_list[csv_file]
        # 长度扩展，最小长度为60.同时要避免超出视频长度。
        seq: pd.core.series.Series = csv.iloc[max(int(start)-padding_left,0):max(int(start)+paddle_right,int(end)),:]
        non_list.append(seq.to_numpy())
        non_index_list.extend([count] * len(seq))
        y.append(0)
        all_time_id.extend([tid for tid in range(len(seq))])
        count += 1
    for row in (root_path/"sc").glob("*.mkv"):
        # 从文件名中寻找信息匹配
        filenames_sc.append(row.name)
        filename_stem,_,start,end = re.search("(EZVZ\d+)-(\d+)-(\d+)-(\d+)",row.stem).groups()
        csv_file = root_path / "csv" / (filename_stem+".csv")
        if csv_file not in csv_list:
            csv_list[csv_file] = pd.read_csv(csv_file,names=['erodesum','diffsum20','erode_sum'])
        csv = csv_list[csv_file]
        # 长度扩展，最小长度为60.同时要避免超出视频长度。
        seq: pd.core.series.Series = csv.iloc[max(int(start)-padding_left,0):max(int(start)+paddle_right,int(end)),:]
        sc_list.append(seq.to_numpy())
        sc_index_list.extend([count] * len(seq))
        y.append(0)
        all_time_id.extend([tid for tid in range(len(seq))])
        count += 1