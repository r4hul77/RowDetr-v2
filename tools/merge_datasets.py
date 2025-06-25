import os
import cv2
import argparse
import shutil
import tqdm

def merge_datasets(dataset1, dataset2, output_dir):
    print(f'Merging datasets {dataset1} and {dataset2}')
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    for split in splits:
        print(f'Merging {split} split')
        imgs_dataset1 = [file for file in os.listdir(os.path.join(dataset1, split, 'images')) if file.endswith('.jpg')]
        imgs_dataset2 = [file for file in os.listdir(os.path.join(dataset2, split, 'images')) if file.endswith('.jpg')]
        imgs_dataset2.sort()
        tot_imgs_d1 = len(imgs_dataset1)
        print(f'Total images in dataset1: {tot_imgs_d1} Copying images from dataset1')
        for img in tqdm.tqdm(imgs_dataset1):
            shutil.copy(os.path.join(dataset1, split, 'images', img), os.path.join(output_dir, split, 'images'))
            shutil.copy(os.path.join(dataset1, split, 'labels', img.replace('.jpg', '.json')), os.path.join(output_dir, split, 'labels'))
        print(f'Copying images from dataset2')
        for i, img in tqdm.tqdm(enumerate(imgs_dataset2)):
            shutil.copy(os.path.join(dataset2, split, 'images', img), os.path.join(output_dir, split, 'images', f'{i+tot_imgs_d1}.jpg'))
            shutil.copy(os.path.join(dataset2, split, 'labels', img.replace('.jpg', '.json')), os.path.join(output_dir, split, 'labels', f'{i+tot_imgs_d1}.json'))


def main():
    parser = argparse.ArgumentParser(description='Merge two datasets')
    parser.add_argument('--dataset1', type=str, help='Path to the first dataset')
    parser.add_argument('--dataset2', type=str, help='Path to the second dataset')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    merge_datasets(args.dataset1, args.dataset2, args.output_dir)

if __name__ == '__main__':
    main()