"""
导出结果
"""
import re
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
# root_path = Path("/media/chenmin/NanShan/misc/CQ-res-orin-split")
root_path = Path("/media/chenmin/NanShan/SRC-res")
# root_path = Path("/media/chenmin/NanShan/result/20340106-CQ-res")

# for i in root_path.rglob("*.mkv"):
#     if i.parent.name == "scratching":
#          print(i.stem)
import pandas as pd
total_df = []
for i in root_path.rglob("*_res.csv"):
    target_ret = pd.read_csv(i)
    if isinstance(target_ret,int):
        continue
    total_df.append(target_ret)
total_df = pd.concat(total_df,ignore_index=True)
total_df = total_df[total_df['behavor']== 1] 
total_df = total_df.groupby('filename', as_index=False).apply(lambda df:df.drop_duplicates("segment"))
print(total_df) 

total_df = total_df.reset_index()

def extract_group(row):
    stem = Path(row).stem
    filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
    return filename_stem

def extract_camera(row):
    stem = Path(row).stem
    filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
    return camera_part
def extract_part(row):
    stem = Path(row).stem
    filename_stem, camera_part, segment = re.search("(\d+)-(\d)_(\d+)", stem).groups()
    return segment

total_df['group'] = total_df['filename'].apply(extract_group)
total_df['camera'] = total_df['filename'].apply(extract_camera)
total_df['part'] = total_df['filename'].apply(extract_part)
total_excel = []
for idx, i in enumerate(CQ_group):
    now = total_df[total_df['group']==i]
    table = now.pivot_table(index='camera',columns='part',values='filename',aggfunc='count',fill_value=0,margins = True, margins_name='Total')
    print(table)
    table['date'] = i
    total_excel.append(table['Total'])