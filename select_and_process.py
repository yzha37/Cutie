from omegaconf import DictConfig, open_dict
from hydra import compose, initialize
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from tqdm import tqdm
from time import perf_counter
from gui.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch
from argparse import ArgumentParser
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2


def generate_mask_1st_frame(video_path):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    cap = cv2.VideoCapture(video_path)

    # Initialize tracker with the first frame and bounding box
    success, frame = cap.read()

    bbox = cv2.selectROI(frame, False)
    input_box = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], dtype=np.int32)
    predictor.set_image(frame)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    cap.release()
    return frame, masks[0]


def process_video(cfg: DictConfig, first_mask):
    # general setup
    torch.set_grad_enabled(False)
    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif cfg['device'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    use_amp = cfg.amp

    # Load the network weights
    print(f'Loading Cutie and weights')
    cutie = CUTIE(cfg).to(device).eval()
    if cfg.weights is not None:
        model_weights = torch.load(cfg.weights, map_location=device)
        cutie.load_weights(model_weights)
    else:
        print('No model weights loaded. Are you sure about this?')

    # Open video
    video = cfg['video']
    if video is None:
        print('No video defined. Please specify!')
        exit()

    print(f'Opening video {video}')
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f'Unable to open video {video}!')
        exit()
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initial mask handling
    # determine if the mask uses 3-channel long ID or 1-channel (0~255) short ID

    num_objects = cfg['num_objects']
    if num_objects is None or num_objects < 1:
        num_objects = len(np.unique(first_mask)) - 1

    processor = InferenceCore(cutie, cfg=cfg)

    # First commit mask input into permanent memory
    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=1)
            pbar.set_description('Commiting masks into permenent memory')

            # load frame matching mask
            _, frame = cap.read()
            if frame is None:
                print(f'No frame captured in video {video}')
                return

            # convert numpy array to pytorch tensor format
            frame_torch = image_to_torch(frame, device=device)

            mask_np = np.array(first_mask)
            mask_torch = index_numpy_to_one_hot_torch(mask_np, num_objects + 1).to(device)

            # the background mask is fed into the model
            prob = processor.step(frame_torch,
                                  mask_torch[1:],
                                  idx_mask=False,
                                  force_permanent=True)

            pbar.update(1)

    # Next start inference on video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset frame reading
    total_process_time = 0
    current_frame_index = 0
    mem_cleanup_ratio = cfg['mem_cleanup_ratio']
    masks = []

    with torch.inference_mode():
        with torch.amp.autocast(device, enabled=(use_amp and device == 'cuda')):
            pbar = tqdm(total=total_frame_count)
            pbar.set_description(f'Processing video {video}')
            while (cap.isOpened()):
                # load frame-by-frame
                _, frame = cap.read()
                if frame is None or current_frame_index > total_frame_count:
                    break

                # timing start
                if 'cuda' in device:
                    torch.cuda.synchronize(device)
                    start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
                    start.record()
                else:
                    a = perf_counter()

                # convert numpy array to pytorch tensor format
                frame_torch = image_to_torch(frame, device=device)
                # propagate only
                prob = processor.step(frame_torch)

                # timing end
                if 'cuda' in device:
                    end.record()
                    torch.cuda.synchronize(device)
                    total_process_time += (start.elapsed_time(end) / 1000)
                else:
                    b = perf_counter()
                    total_process_time += (b - a)

                # save mask
                masks.append(torch.argmax(prob, dim=0))

                check_to_clear_non_permanent_cuda_memory(processor=processor,
                                                         device=device,
                                                         mem_cleanup_ratio=mem_cleanup_ratio)

                current_frame_index += 1
                pbar.update(1)

    pbar.close()
    cap.release()  # Release the video capture object

    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )
    print(f'Total processing time: {total_process_time}')
    print(f'Total processed frames: {current_frame_index}')
    print(f'FPS: {current_frame_index / total_process_time}')
    print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}'
          ) if device == 'cuda' else None
    print(
        '------------------------------------------------------------------------------------------------------------------------------------------------'
    )
    return masks


def check_to_clear_non_permanent_cuda_memory(processor: InferenceCore, device, mem_cleanup_ratio):
    if 'cuda' in device:
        if mem_cleanup_ratio > 0 and mem_cleanup_ratio <= 1:
            info = torch.cuda.mem_get_info()

            global_free, global_total = info
            global_free /= (2**30)  # GB
            global_total /= (2**30)  # GB
            global_used = global_total - global_free
            #mem_ratio = round(global_used / global_total * 100)
            mem_ratio = global_used / global_total
            if mem_ratio > mem_cleanup_ratio:
                print(f'GPU cleanup triggered: {mem_ratio} > {mem_cleanup_ratio}')
                processor.clear_non_permanent_memory()
                torch.cuda.empty_cache()


def apply_colored_mask(frame, mask, color):
    # Create a mask where white parts (255) are True
    mask_boolean = mask == mask.max()

    # Create an empty colored mask
    colored_mask = np.zeros_like(frame)

    # Apply color only to the mask's white parts
    colored_mask[mask_boolean] = color

    # Combine the colored mask with the frame
    frame[mask_boolean] = cv2.addWeighted(frame[mask_boolean], 0.5, colored_mask[mask_boolean], 0.5, 0)

    return frame


def store_first_mask(frame, mask, output_name):
    color = [0, 255, 0]  # Green color in BGR format
    masked_frame = apply_colored_mask(frame, mask, color)
    cv2.imwrite(output_name, masked_frame)


def generate_video(video_path, masks, output_path):
    # Read the original video
    cap = cv2.VideoCapture(video_path)

    # Video writer to create a new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Specify a single color for all masks
    color = [0, 255, 0]  # Green color in BGR format

    # Iterate through video frames and apply masks
    for output_mask in masks:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_colored_mask(frame, output_mask, color)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='Video file.', default='examples/videos/lalaland_trip2.mov')
    parser.add_argument(
        '-m',
        '--mask_dir',
        help=
        'Directory with mask files. Must be named with with corresponding video frame number syntax [07d].',
        default='examples/mask')
    parser.add_argument('-o',
                        '--output_dir',
                        help='Directory where processed mask files will be saved.',
                        default='output')
    parser.add_argument('-d',
                        '--device',
                        help='Target device for processing [cuda, cpu].',
                        default='cpu')
    parser.add_argument(
        '--mem_every',
        help='How often to update working memory; higher number speeds up processing.',
        type=int,
        default='10')
    parser.add_argument(
        '--max_internal_size',
        help=
        'maximum internal processing size; reducing this speeds up processing; -1 means no resizing.',
        type=int,
        default='480')
    parser.add_argument(
        '--mem_cleanup_ratio',
        help=
        'How often to clear non permanent GPU memory; when ratio of GPU memory used is above given mem_cleanup_ratio [0;1] then cleanup is triggered; only used when device=cuda.',
        type=float,
        default='-1')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # input arguments
    args = get_arguments()

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path="cutie/config", job_name="process_video")
    cfg = compose(config_name="video_config")

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            cfg[k] = v

    first_frame, first_mask = generate_mask_1st_frame(cfg['video'])
    output_image_name = cfg['video'].split('/')[-1].split('.')[0] + '_first_mask.png'
    store_first_mask(first_frame, first_mask, 'output/' + output_image_name)

    output_masks = process_video(cfg, first_mask)
    output_video_name = cfg['video'].split('/')[-1].split('.')[0] + '_output.mp4'
    generate_video(cfg['video'], output_masks, 'output/' + output_video_name)
