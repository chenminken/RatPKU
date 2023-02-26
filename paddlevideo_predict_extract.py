"""
从paddle video 的predict.py 生成的日志文件中提取出视频的预测结果
日志文件示例：
Current video file: 2023-2-22T22_12_50_9-30-14769-15579_48.mkv
        top-1 class: [3]
        top-1 score: [0.80324507]
打印日志
"""
import os
import argparse

def read_log_file(filename):
    # 读取日志文件
    print(f'log file path: {filename}')
    with open(filename, 'r', encoding='UTF-8') as f:
        # 读取所有行，行数是3的倍数。每三行是一个视频的预测结果。
        lines = f.readlines()
        # 保存每个视频的预测结果
        video_predict_result = []
        # 保存每个视频的预测结果的类别
        video_predict_class = []
        # 保存每个视频的预测结果的分数
        video_predict_score = []
        for i in range(0, len(lines), 3):
            # 保存每个视频的预测结果, 提取出视频的文件名
            video_predict_result.append(lines[i].split(':')[-1].strip())
            # 保存每个视频的预测结果的类别, 提取出类别,去掉中括号。
            video_predict_class.append(lines[i+1].split(':')[-1].strip()[1:-1])
            # 保存每个视频的预测结果的分数，去掉中括号
            video_predict_score.append(lines[i+2].split(':')[-1].strip()[1:-1])
    return video_predict_result, video_predict_class, video_predict_score

def move_file(src_file, dst_file):
    # 移动文件
    if os.path.exists(src_file):
        os.rename(src_file, dst_file)
    else:
        print(f'file {src_file} not exist')

# 将预测结果中的文件移动到类别标签的文件夹中
def move_file_to_class_folder(video_predict_result, video_predict_class, video_predict_score, src_folder, dst_folder):

    for i in range(len(video_predict_result)):
        # 提取视频的文件名
        video_name = video_predict_result[i].split('/')[-1]
        # 提取视频的预测类别
        video_class = video_predict_class[i]
        # 提取视频的预测分数
        video_score = video_predict_score[i]
        # 源文件路径
        src_file = os.path.join(src_folder, video_name)
        # 目标文件夹路径， 不与入参dst_folder相同，避免覆盖
        dst_folder_copy = os.path.join(dst_folder, video_class)
        # 目标文件路径
        dst_file = os.path.join(dst_folder_copy, video_name)
        # 如果目标文件夹不存在，则创建
        if not os.path.exists(dst_folder_copy):
            os.makedirs(dst_folder_copy)
        # 移动文件
        move_file(src_file, dst_file)
        
    
# 从命令行中读取参数
def parse_args():
    parser = argparse.ArgumentParser(description='extract predict result from paddle video predict log file')
    parser.add_argument('--log_path', type=str, default='log.txt',
                        help='the path of log file')
    # 保存文件夹的位置
    parser.add_argument('--save_folder', type=str, default='D:\\研究生所有代码\\')
    parser.add_argument('--save_path', type=str, default='predict_result.txt',
                        help='the path of save file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    log_path = args.log_path
    save_path = args.save_path
    save_folder = args.save_folder
    # 读取日志文件
    video_predict_result, video_predict_class, video_predict_score = read_log_file(log_path)
    # 保存预测结果
    move_file_to_class_folder(video_predict_result, video_predict_class, video_predict_score, save_folder, 'video_class_ear2')
    with open(save_path, 'w+', encoding='UTF-8') as f:
        for i in range(len(video_predict_result)):
            f.write(video_predict_result[i] + ',' + video_predict_class[i] + ',' + video_predict_score[i] + '\n')

if __name__ == '__main__':
    main()