import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import argparse
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Interpolation for image sequence based on set of timestamps provided by lidar scans'
)
parser.add_argument('--image_dir', required=True)
parser.add_argument('--lidar_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

args = parser.parse_args()

try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    from model.RIFE_HD import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v1.x HD model")
if not hasattr(model, 'version'):
    model.version = 0
model.eval()
model.device()


def get_prefix(file_name):
    return file_name[:file_name.rfind('-') + 1]


def get_suffix(file_name):
    return os.path.splitext(file_name)[1]


all_image_names = os.listdir(args.image_dir)

file_name_prefix = get_prefix(all_image_names[0])
file_name_suffix = get_suffix(all_image_names[0])
image_stamps = np.array([
    float(name[
        len(file_name_prefix):-len(file_name_suffix)
    ]) for name in all_image_names
])

lidar_name_prefix_len = os.listdir(args.lidar_dir)[0].rfind('_') + 1
lidar_stamps = sorted([
    name[
        lidar_name_prefix_len:-len(file_name_suffix)
    ] for name in os.listdir(args.lidar_dir)
])


def find_closest_before(image_stamps, stamp):
    diff = image_stamps - stamp
    valid_idx = np.where(diff < 0)[0]
    idx = valid_idx[diff[valid_idx].argmax()]
    return idx, image_stamps[idx]


def find_closest_after(image_stamps, stamp):
    diff = image_stamps - stamp
    valid_idx = np.where(diff > 0)[0]
    idx = valid_idx[diff[valid_idx].argmin()]
    return idx, image_stamps[idx]


def infer_image(img0, img1, ratio):
    if model.version >= 3.9:
        result = model.inference(img0, img1, ratio)
    else:
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + args.rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - args.rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(args.rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = (img0_ratio + img1_ratio) / 2
                if ratio - (args.rthreshold / 2) <= middle_ratio <= ratio + (args.rthreshold / 2):
                    break
                if ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        result = middle

    return result


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    # img = cv2.resize(img, (448, 256))
    img = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    shape = img.shape
    ph = ((shape[2] - 1) // 64 + 1) * 64
    pw = ((shape[3] - 1) // 64 + 1) * 64
    padding = (0, pw - shape[3], 0, ph - shape[2])
    img = F.pad(img, padding)
    return img, shape[0], shape[1], shape[2], shape[3]


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for desired_stamp_str in lidar_stamps:
    desired_stamp = float(desired_stamp_str)
    idx_before, stamp_before = find_closest_before(image_stamps, desired_stamp)
    idx_after, stamp_after = find_closest_after(image_stamps, desired_stamp)

    path0 = os.path.join(args.image_dir, all_image_names[idx_before])
    path1 = os.path.join(args.image_dir, all_image_names[idx_after])

    img0 = cv2.imread(path0, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    img0, n, c, h, w = read_image(path0)
    img1, n, c, h, w = read_image(path1)

    ratio = (desired_stamp - stamp_before) / (stamp_after - stamp_before)
    print(f'Infering {stamp_before} < {desired_stamp} < {stamp_after} ({ratio})')
    image = infer_image(img0, img1, ratio)

    out_path = os.path.join(
        args.output_dir,
        f'{file_name_prefix}_{desired_stamp_str}{file_name_suffix}'
    )
    cv2.imwrite(
        out_path,
        (image[0]*255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
    )
