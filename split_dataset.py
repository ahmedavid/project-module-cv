"""Split generated floor plan data into train / val / test sets."""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('-i', '--input-dir', type=str, default='data',
                        help='Input directory containing images/ and masks/')
    parser.add_argument('-o', '--output-dir', type=str, default='data',
                        help='Output directory for split (default: same as input)')
    parser.add_argument('--train', type=int, default=1600)
    parser.add_argument('--val', type=int, default=200)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    images = sorted((inp / 'images').glob('*.png'))
    masks = sorted((inp / 'masks').glob('*.png'))
    assert len(images) == len(masks), 'Image/mask count mismatch'

    total_needed = args.train + args.val + args.test
    assert len(images) >= total_needed, (
        f'Need {total_needed} samples but only found {len(images)}'
    )

    # Shuffle deterministically
    indices = list(range(len(images)))
    random.Random(args.seed).shuffle(indices)

    splits = {
        'train': indices[:args.train],
        'val': indices[args.train:args.train + args.val],
        'test': indices[args.train + args.val:args.train + args.val + args.test],
    }

    for split_name, split_indices in splits.items():
        img_out = out / split_name / 'images'
        msk_out = out / split_name / 'masks'
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        for new_idx, orig_idx in enumerate(split_indices):
            fname = f'floor_plan_{new_idx:04d}.png'
            shutil.copy2(images[orig_idx], img_out / fname)
            shutil.copy2(masks[orig_idx], msk_out / fname)

        print(f'  {split_name:5s}: {len(split_indices)} samples -> {img_out.parent}')

    print('Done!')


if __name__ == '__main__':
    main()
