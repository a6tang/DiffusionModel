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
import argparse

def get_base_filename(file_path):
    base_name = os.path.basename(file_path)
    # Split on first period only
    return base_name.split('.')[0]

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
        crop_files = _list_image_files_recursively(os.path.join(data_dir, 'crops', 'train' if is_train else 'test'))
        mask_file = _list_image_files_recursively(os.path.join(data_dir, 'masks', 'train' if is_train else 'test'))
        target_file = _list_image_files_recursively(os.path.join(data_dir, 'targets', 'train' if is_train else 'test'))
        
    print("Len of Dataset:", len(crop_files))
    print("Len of mask_file:", len(mask_file))
    print("Len of target_file:", len(target_file))
    # print("Len of crop_file:", len(crop_file))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        crops=crop_files,
        maskes=mask_file,
        targetes=target_file,
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train,
        tokenizer=tokenizer,
        args=args,
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



class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        crops,
        maskes=None,
        targetes=None,
        random_crop=False,
        random_flip=True,
        is_train=True,
        tokenizer=None,
        args=None,
        snr = 10,
        vae = None
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_crops = crops
        self.local_maskes = maskes
        self.local_target = targetes
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.args = args
        self.tokenizer = tokenizer
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        # self.control_image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        # )

    def __len__(self):
        return len(self.local_crops)

    def __getitem__(self, idx):
        path_crops = self.local_crops[idx]
        with bf.BlobFile(path_crops, "rb") as f:
            crop_image = Image.open(f)
            crop_image.load()
        #pil_image = pil_image.convert("RGB")
        
        #print ('image.shape',image.shape)
        
        path_mask = self.local_maskes[idx]
        with bf.BlobFile(path_mask, "rb") as f:
            mask_image = Image.open(f)
            mask_image.load()
        
        path_target = self.local_target[idx]
        with bf.BlobFile(path_target, "rb") as f:
            target_image = Image.open(f)
            target_image.load()

        crop_image = self.control_image_processor.preprocess(crop_image, 
                                                                    height=self.resolution, 
                                                                    width=self.resolution
                                                                    ).to(dtype=torch.float32).squeeze()
        mask_image = self.control_image_processor.preprocess(mask_image, 
                                                                    height=self.resolution, 
                                                                    width=self.resolution
                                                                    ).to(dtype=torch.float32).squeeze()
        target_image = self.control_image_processor.preprocess(target_image, 
                                                                    height=self.resolution, 
                                                                    width=self.resolution
                                                                    ).to(dtype=torch.float32).squeeze()
        control_image = torch.cat([crop_image, mask_image], dim=1)
        print ('control_image.shape',control_image.shape)
        return control_image, target_image
if __name__ == "__main__":
    args = parse_args()
    main(args)