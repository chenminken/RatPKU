from time import time
import argparse
from typing import Union
import numpy as np
from datetime import datetime
import logging
from rich.pretty import pprint
import os
# from sleap.io.dataser import labels
# import sleap
# from sleap.nn.data.pipelines import (
#     Provider,
#     Pipeline,
#     LabelsReader,
#     VideoReader,
#     Normalizer,
#     Resizer,
#     Prefetcher,
#     InstanceCentroidFinder,
#     KerasModelPredictor,
# )
from pathlib import Path
import pickle
import shutil
import pandas as pd

logger = logging.getLogger(__name__)


inter = 10
intra = 40
area_min = 20
area_max = 200
gray_thres = 60

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
    parser.add_argument("--suffix", type=str,default="MP4")

    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default="",
        help=(
            "Path to data to predict on. This can be a labels (.slp) file or any "
            "supported video format."
        ),
    )
    parser.add_argument(
        "--only_clip",
        action='store_true',
        default=False,
        help='only clip video')
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
    # parser.add_argument(
    #     "--frames",
    #     type=str,
    #     default="",
    #     help=(
    #         "List of frames to predict when running on a video. Can be specified as a "
    #         "comma separated list (e.g. 1,2,3) or a range separated by hyphen (e.g., "
    #         "1-3, for 1,2,3). If not provided, defaults to predicting on the entire "
    #         "video."
    #     ),
    # )
    parser.add_argument(
        "--only-labeled-frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on user labeled frames when running on labels dataset. "
            "This is useful for generating predictions to compare against ground truth."
        ),
    )
    parser.add_argument(
        "--only-suggested-frames",
        action="store_true",
        default=False,
        help=(
            "Only run inference on unlabeled suggested frames when running on labels "
            "dataset. This is useful for generating predictions for initialization "
            "during labeling."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "The output filename to use for the predicted data. If not provided, "
            "defaults to '[data_path].predictions.slp'."
        ),
    )
    parser.add_argument(
        "--no-empty-frames",
        action="store_true",
        default=False,
        help=(
            "Clear any empty frames that did not have any detected instances before "
            "saving to output."
        ),
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
        "--first-gpu",
        action="store_true",
        help="Run inference on the first GPU, if available.",
    )
    device_group.add_argument(
        "--last-gpu",
        action="store_true",
        help="Run inference on the last GPU, if available.",
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
    parser.add_argument(
        "--max_edge_length_ratio",
        type=float,
        default=0.25,
        help="The maximum expected length of a connected pair of points "
        "as a fraction of the image size. Candidate connections longer "
        "than this length will be penalized during matching. "
        "Only applies to bottom-up (PAF) models.",
    )
    parser.add_argument(
        "--dist_penalty_weight",
        type=float,
        default=1.0,
        help="A coefficient to scale weight of the distance penalty. Set "
        "to values greater than 1.0 to enforce the distance penalty more strictly. "
        "Only applies to bottom-up (PAF) models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Number of frames to predict at a time. Larger values result in faster "
            "inference speeds, but require more memory."
        ),
    )
    parser.add_argument(
        "--open-in-gui",
        action="store_true",
        help="Open the resulting predictions in the GUI when finished.",
    )
    parser.add_argument(
        "--peak_threshold",
        type=float,
        default=0.2,
        help="Minimum confidence map value to consider a peak as valid.",
    )

    # Deprecated legacy args. These will still be parsed for backward compatibility but
    # are hidden from the CLI help.
    parser.add_argument(
        "--labels",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--topdown.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bottomup.peak_threshold",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--topdown.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bottomup.batch_size",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    return parser

def run_preprocess(src : Path, workdir : Path,exist_ok=True):
    """
    预处理视频，包括：
    1. 计算视频的帧间差分
    2. 过滤帧间差分，得到分段信息
    3. 根据分段信息，将视频分段
    返回分段信息和每一段包含哪些帧的列表
    Args:
    - src 视频文件的Path对象
    Returns:
    - segment_df 分段信息
    - frame_list 每一段包含哪些帧的列表
    """
    from utils import cal_video,filter_points,filter,clipSelected
    dst_csv = str(workdir / ("{}.csv".format(src.stem)))
    print("dest csv",dst_csv)
    if not(exist_ok and Path(dst_csv).is_file()):
        cal_video(str(src), dst_csv)
    pass_points, diff_df = filter_points(dst_csv)
    #获得分段信息，包含st,end, info(avg,min,max,count)
    segment_df = filter(pass_points, inter, intra, diff_df) 
    dst_prefix = str(workdir / ("{}".format(src.stem) + "-{}-{}-{}.mkv"))
    clipSelected(src=str(src),dst_prefix=dst_prefix,points=segment_df)
    frame_list = []
    for i in segment_df:
        frame_list += list(range(i['st'], i['end']))
    logger.info(f"frame_list{len(frame_list)}")
    return frame_list,segment_df


def main(args: list = None):
    """Entrypoint for `mouse-sleap` CLI for running inference.

    Args:
        args: A list of arguments to be passed into mouse-sleap.
    """

    #第一步：计算长视频的差分帧信息，存放到‘.csv’文件中，sleap信息存放在‘.h5’文件中
    t0 = time()
    start_timestamp = str(datetime.now())
    print("Started inference at:", start_timestamp)

    # Setup CLI.
    parser = _make_cli_parser()
    args, _ = parser.parse_known_args(args=args)
    print("Args:")
    pprint(vars(args))
    print()

    output_path = args.output
    print(output_path)
    # Setup output device
    root_path = Path(args.data_path)
    output_path = Path(output_path)
    video_suffix = args.suffix
    recursive = args.recursive
    if root_path.is_file():
        video_list = [root_path]
    else:
        video_list = root_path.rglob(f"*.{video_suffix}") if recursive else root_path.glob(f"*.{video_suffix}")


    # Setup devices.
#    if args.cpu or not sleap.nn.system.is_gpu_system():
#        sleap.nn.system.use_cpu_only()
#    else:
#        if args.first_gpu:
#            sleap.nn.system.use_first_gpu()
#        elif args.last_gpu:
#            sleap.nn.system.use_last_gpu()
#        else:
#            if args.gpu == "auto":
#                free_gpu_memory = sleap.nn.system.get_gpu_memory()
#                if len(free_gpu_memory) > 0:
#                    gpu_ind = np.argmax(free_gpu_memory)
#                    logger.info(
#                        f"Auto-selected GPU {gpu_ind} with {free_gpu_memory} MiB of "
#                        "free memory."
#                    )
#                else:
#                    logger.info(
#                        "Failed to query GPU memory from nvidia-smi. Defaulting to "
#                        "first GPU."
#                    )
#                    gpu_ind = 0
#            else:
#                gpu_ind = int(args.gpu)
#            sleap.nn.system.use_gpu(gpu_ind)
#    sleap.disable_preallocation()
#    # Load model
#    if args.models is not None:
#        predictor: Predictor = sleap.load_model(
#            args.models,
#            peak_threshold=args.peak_threshold,
#            batch_size=args.batch_size,
#            refinement="integral",
#            progress_reporting=args.verbosity,
#        )
#    else:
#        predictor = sleap.load_model("weights/221113_110806.single_instance.n=697")
#
    # Extract suggestion frame by frame difference algorithm.
    """
    对于每一个视频文件，首先计算差分帧，然后根据差分帧的信息，进行潜在行为片段提取
    """
    for video_filename in video_list:
        workdir =  video_filename.parent
        workdir = Path(str(workdir).replace(str(root_path),str(output_path)))
        workdir.mkdir(exist_ok=True,parents=True)
        output_stem = str(workdir/video_filename.stem)
        # frame_list表示哪些帧会被提取
        frame_list,info = run_preprocess(video_filename, workdir)
        logger.info(f"{len(frame_list)}")
        # logger.info(frame_list)
        if len(frame_list) == 0:
            logger.warning(f"Current file '{video_filename}' has not frame difference")
            continue
        # if Path(output_stem+".slp").is_file():
        #     logger.warn(f"Current file '{video_filename}' has already inferenced")
        #     continue
#        provider = VideoReader.from_filepath(
#            filename=str(video_filename), example_indices=frame_list
#        )
#        labels_pr: Union[list[dict[str, np.ndarray]], sleap.Labels] = predictor.predict(provider)
#        labels_pr.save(output_stem)
        # postprocess
#        from sleap.info.write_tracking_h5 import main as write_analysis
#        video_callback = Labels.make_video_callback(str(video_filename))
#        labels: Labels = Labels.load_file(output_stem+".slp", video_search=video_callback)
#        write_analysis(
#                labels,
#                output_path=output_stem+'.h5',
#                labels_path=output_stem+".slp",
#                all_frames=True,
#        )
    #第二步：提取‘.h5’中的sleap识别信息，并输出到‘_re.csv’中
    if args.only_clip: #只想提取潜在行为片段，不想进行后续识别
        return
    # print('begin 2')
    # from utils import export_csv
    # fileList = os.listdir(output_path)
    # for file in fileList:
    #     path = os.path.join(output_path, file)
    #     if(path[-2:] == 'h5'):
    #         export_csv(path, path+'_re.csv')

    #第三步：将‘_re.csv’中的hind信息拼接到‘.csv’中，结果保存到‘_withHind.csv’中
    # print('begin 3')
    # from utils import fillData
    # fileList = os.listdir(output_path)
    # for file in fileList:
    #     path = os.path.join(output_path, file)
    #     if(path[-2:] == 'h5'):
    #         print(path)
    #         if Path(path[:-3]+'_withHind.csv').exists():
    #             print('Already exit, ignore it. ', path[:-3]+'_withHind.csv')
    #             continue
    #         tar_csv = pd.read_csv(path[:-3]+'.csv',)
    #         res_csv = pd.read_csv(path[:-3]+'_re.csv')
    #         print(list(res_csv.columns))
    #         # if 'hind part_x' not in list(res_csv.columns):
    #         #     continue
    #         hind_x = list(np.array(res_csv['hind part_x']))
    #         hind_y = list(np.array(res_csv['hind part_y']))
            
    #         hind_x = fillData(hind_x)
    #         hind_y = fillData(hind_y)
    #         # print(tar_csv.columns)
    #         tar_csv.insert(4, hind_x[0], np.append(hind_x[1:], [0] * (len(tar_csv) - len(hind_x) + 1)))
    #         tar_csv.insert(5, hind_y[0] + 0.1, np.append(hind_y[1:], [0] * (len(tar_csv) - len(hind_y) + 1)))
    #         outputpath= path[:-3]+'_withHind.csv'
    #         tar_csv.to_csv(outputpath,sep=',',index=False,header=True)

    #第四步：调用模型识别是否为抓挠(前提是模型已经训练好了) 不知道脚本里面输入怎么写就暂时写死
    # print('begin 4')
    # from utils import predict
    # with open(Path('./dashu-v1-20221205.pickle'), 'rb') as fw:
    #     clf=pickle.load(fw)
    #     event_dicts = predict(output_path,True, clf)
    #     print(event_dicts)

    #第五步：创建ScratchingEvent列表并返回，将结果输出在‘文件名_res.csv’中，其中没行信息代表一个差分帧识别到的动作片段
    #每一行的信息包含：id,开始时间，结束时间，是否是抓挠（1是0不是），如果是抓挠抓挠哪一侧（-1为识别到，1为左侧，0为右侧）
    # from utils import ScratchingEvent, judge_side
    # for key in event_dicts:
    #     sc_list = []
    #     info = event_dicts[key]
    #     video_info = pd.read_csv(str(output_path) + '/' + key.split('_')[0] + '_re.csv')
    #     i = 0
    #     seg = info['seg'][0]
    #     count_sc = 0
    #     count_non = 0
    #     while True:
    #         if info['seg'][i] == seg:
    #             if info['predict'][i] == 1:
    #                 count_sc += 1
    #             else:
    #                 count_non += 1
    #         else:
    #             event_piece = ScratchingEvent()
    #             if count_sc >= count_non:
    #                 event_piece.type = 1
    #             else:
    #                 event_piece.type = 0
    #             event_piece.start_frame = int(seg.split('-')[0])
    #             event_piece.end_frame = int(seg.split('-')[1])
    #             event_piece.start_time = event_piece.start_frame/120
    #             event_piece.end_time = event_piece.end_frame/120
    #             if event_piece.type == 1:
    #                 event_piece.side = judge_side(event_piece, video_info)
    #             sc_list.append(event_piece)
    #             if info['predict'][i] == 1:
    #                 count_sc = 1
    #                 count_non = 0
    #             else:
    #                 count_sc = 0
    #                 count_non = 1
    #             seg = info['seg'][i]
    #         i += 1
    #         if i == info.size/2:
    #             event_piece = ScratchingEvent()
    #             if count_sc >= count_non:
    #                 event_piece.type = 1
    #             else:
    #                 event_piece.type = 0
    #             event_piece.start_frame = int(seg.split('-')[0])
    #             event_piece.end_frame = int(seg.split('-')[1])
    #             event_piece.start_time = event_piece.start_frame/120
    #             event_piece.end_time = event_piece.end_frame/120
    #             if event_piece.type == 1:
    #                 event_piece.side = judge_side(event_piece, video_info)
    #             sc_list.append(event_piece)
    #             break
    #     pd.DataFrame([i.__dict__() for i in sc_list]).to_csv(str(output_path) + '/' + key.split('_')[0] + '_res.csv')
    

if __name__ == "__main__":
    main()
