# 写一个程序来读取文件夹的所有视频，然后将视频拆分成多个定长的子视频文件，不是图片文件，子视频的长度由参数决定，然后将子视频保存到另一个文件夹中
# 作者 陈闽
# 时间：2021/2/21

import os
import cv2
import numpy as np
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_path', type=str, default='./data/videos',
                        help='path to the video folder')
    parser.add_argument('--save_path', type=str, default='./data/frames',
                        help='path to the save folder')
    parser.add_argument('--frame_length', type=int, default=24,
                        help='the length of each frame')

    parser.add_argument('--frame_format', type=str, default='mkv',
                        help='the format of each frame')
    parser.add_argument('--frame_prefix', type=str, default='frame',
                        help='the prefix of each frame')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    video_path = args.video_path
    save_path = args.save_path
    frame_length = args.frame_length
    frame_format = args.frame_format

    # 只读取文件夹中的视频文件
    video_list = [video_name for video_name in os.listdir(video_path) if video_name.split('.')[-1] in ['mp4', 'avi', 'mkv']]
    video_list.sort()
    # 创建save_path文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for video_name in video_list:
        # 子视频的序号，从0开始
        sub_video_index = 0
        video_path_name = os.path.join(video_path, video_name)
        # 命名需要保持后缀名一致，然后在文件名中安装子视频的个数进行编号
        save_path_name = os.path.join(save_path, video_name.split('.')[0] + f'_{sub_video_index}.' + frame_format)
        # 创建videocapture对象, 需要转义字符，否则会报错
        cap = cv2.VideoCapture(video_path_name.replace('\\', '\\\\'))
        # 获取cap的宽高
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        # 如果文件已经存在，跳过
        if os.path.exists(save_path_name):
            continue
        # 创建保存视频的videowriter对象, 保存的视频格式为mkv
        out_vid = cv2.VideoWriter(save_path_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (frame_width, frame_height))
        while True:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                # 依次保存每一帧到out_vid中
                out_vid.write(frame)
                # 当 frame_count 是 frame_length的整数时，重新创造一个out_vid对象
                if frame_count % frame_length == 0 and frame_count != 0:
                    out_vid.release()
                    sub_video_index += 1
                    save_path_name = os.path.join(save_path, video_name.split('.')[0] + f'_{sub_video_index}.' + frame_format)
                    if os.path.exists(save_path_name):
                        continue
                    out_vid = cv2.VideoWriter(save_path_name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (frame_width, frame_height))

            else:
                # 删除最后一个子视频，因为最后一个子视频的帧数不足frame_length
                out_vid.release()
                print(f'video {video_name} has been processed')
                os.remove(save_path_name)
                break
        cap.release()

# 运行程序
if __name__ == '__main__':
    main()
