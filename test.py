import argparse
import torch
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from image_datasets2 import load_data

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to test load_data function")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    return parser.parse_args(input_args)

def test_load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, use_fast=False)
    
    # Load the VAE model for potential preprocessing
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )

    # Load data using load_data
    data_loader = load_data(
        dataset_mode='underwater',
        data_dir='final_stacked_dataset_5',
        batch_size=args.train_batch_size,
        image_size=512,
        tokenizer=tokenizer,
        args=args,
        class_cond=True,
        deterministic=False,
        random_crop=False,
        random_flip=False,
        is_train=True,
        vae=vae,
        add_snr=True,
    )

    # Iterate over a few batches for testing
    for batch_idx, (control_image, target_image) in enumerate(data_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Batch Shape: {control_image.shape}")
        print(f"  taget shape: {target_image.shape}")
        if batch_idx >= 2:  # Stop after a few batches
            break

if __name__ == "__main__":
    args = parse_args()
    test_load_data(args)
