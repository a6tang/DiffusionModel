import os
import math
import random
import string

from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from diffusers.image_processor import VaeImageProcessor
import sys
import torch
import pickle
import cv2
from qam2 import qam16ModulationString,qam16ModulationTensor

def get_base_filename(file_path):
    base_name = os.path.basename(file_path)
    # Split on first period only
    return base_name.split('.')[0]

def snr_to_noise_probability(snr):
    """
    将SNR转换为噪音概率。
    假设SNR为dB, 噪音概率为1 / (1 + SNR)。
    """
    linear_snr = 10 ** (snr / 10)
    noise_probability = 1 / (1 + linear_snr)
    return noise_probability

def add_noise_to_text(text, noise_probability):
    """
    基于噪音概率对文本进行加噪音。
    这里我们用随机字符替换一定比例的字符来模拟噪音。
    """
    noisy_text = []
    for char in text:
        if random.random() < noise_probability:
            # 替换为随机字符
            noisy_char = random.choice(string.ascii_letters + string.punctuation + string.digits + " ")
            noisy_text.append(noisy_char)
        else:
            noisy_text.append(char)
    return ''.join(noisy_text)

def load_data(
    *,
    dataset_mode,
    data_dir,
    batch_size,
    image_size,
    tokenizer,
    args,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
    is_train=True,
    one_hot_label=True,
    limit_human_robot = False,
    add_snr = False,
    snr = None,
    vae = None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == 'underwater':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'crops', 'train' if is_train else 'test'))
        labels_file = _list_image_files_recursively(os.path.join(data_dir, 'masks', 'train' if is_train else 'test'))
        depth_file = _list_image_files_recursively(os.path.join(data_dir, 'targets', 'train' if is_train else 'test'))
        # crop_file = _list_image_files_recursively(os.path.join(data_dir, 'mask_images', 'train' if is_train else 'test'))
        # qa_dict = pickle.load(open(os.path.join(data_dir, 'qa_dataset', 'train_all.pkl' if is_train else 'test.pkl'), 'rb'))
        if limit_human_robot:
            human_robot_files = _list_image_files_recursively(os.path.join(data_dir, 'images_objective', 'train' if is_train else 'test'))
            human_robot_files = [os.path.splitext(os.path.basename(x))[0] for x in human_robot_files]
            def match_files(file_list):
                return [x for x in file_list if get_base_filename(x) in human_robot_files]
            all_files = match_files(all_files)
            labels_file = match_files(labels_file)
            depth_file = match_files(depth_file)
            # crop_file = match_files(crop_file)
    print("Len of Dataset:", len(all_files))
    print("Len of labels_file:", len(labels_file))
    print("Len of depth_file:", len(depth_file))
    # print("Len of crop_file:", len(crop_file))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        classes=labels_file,
        depthes=depth_file,
        # crops=crop_file,
        # qa_dict = qa_dict,
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train,
        tokenizer=tokenizer,
        args=args,
        add_snr = add_snr,
        snr = snr,
        vae = vae
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True, pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True, pin_memory=True
        )
    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp",'npy']:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def standardize(x):
    mean = x.mean()
    std = x.std()
    standardized_x = (x - mean) / std
    return standardized_x, mean, std

# 逆标准化函数
def destandardize(x, mean, std):
    return x * std + mean

class Channels():
    def __init__(self,device='cuda'):
        self.device = device

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1])
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1])
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]])
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1])
        H_imag = torch.normal(mean, std, size=[1])
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]])
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        depthes=None,
        # crops = None,
        # qa_dict = None,
        random_crop=False,
        random_flip=True,
        is_train=True,
        tokenizer=None,
        args=None,
        add_noise=False,
        add_snr = False,
        snr = 10,
        vae = None
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths
        self.local_class = classes
        self.local_depth = depthes
        # self.local_crops = crops
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.args = args
        self.tokenizer = tokenizer
        # self.qa_dict = qa_dict
        self.add_snr = add_snr
        self.snr = snr
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.channels = Channels()

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        #pil_image = pil_image.convert("RGB")
        
        #print ('image.shape',image.shape)
        

        pil_image_small = self.control_image_processor.preprocess(pil_image_small, 
                                                                    height=self.resolution, 
                                                                    width=self.resolution
                                                                    ).to(dtype=torch.float32).squeeze()
        
if __name__ == "__main__":
    args = parse_args()
    main(args)