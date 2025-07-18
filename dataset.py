import os
import cv2
from PIL import Image
import torch
from torchvision import transforms  # Import transforms here
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

from utils import one_hot_to_image, image_to_class_index, class_index_to_image


class SatelliteImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, aug_image_dir=None, aug_mask_dir=None, num_classes=6, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmented_image_dir = aug_image_dir
        self.augmented_mask_dir = aug_mask_dir
        self.num_classes = num_classes
        self.transform = transform

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        if aug_image_dir and aug_mask_dir:
            self.augmented_image_filenames = sorted(os.listdir(self.augmented_image_dir))
            self.augmented_mask_filenames = sorted(os.listdir(self.augmented_mask_dir))
        else:
            self.augmented_image_filenames = []
            self.augmented_mask_filenames = []

        self.color_map = {
            (60, 16, 152): 0,  # building
            (132, 41, 246): 1,  # land
            (110, 193, 228): 2,  # road
            (254, 221, 58): 3,  # vegetation
            (226, 169, 41): 4,  # water
            (155, 155, 155): 5,  # unlabeled / unknown
        }

        self.mask_labels = ['building', 'land', 'road', 'vegetation', 'water', 'unlabeled']

        assert len(self.image_filenames) == len(self.mask_filenames)
        assert len(self.augmented_image_filenames) == len(self.augmented_mask_filenames)

    def __getitem__(self, i):
        if self.augmented_image_dir and self.augmented_mask_dir:
            image_path = os.path.join(self.augmented_image_dir, self.augmented_image_filenames[i])
            mask_path = os.path.join(self.augmented_mask_dir, self.augmented_mask_filenames[i])
        else:
            image_path = os.path.join(self.image_dir, self.image_filenames[i])
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[i])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB before transformations

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        resized_mask = cv2.resize(mask, (512, 512))
        # (c, h, w)
        mask = torch.tensor(resized_mask, dtype=torch.uint8).permute(2, 0, 1)

        mask = image_to_class_index(mask, self.color_map)
        mask = mask.long()

        return image, mask

    def __len__(self):
        return len(self.augmented_image_filenames)


    
    def augment(self, dest_image_dir, dest_mask_dir):
        self.augmented_image_dir = dest_image_dir
        self.augmented_mask_dir = dest_mask_dir

        for i in range(5):
            for image_filename, mask_filename in zip(self.image_filenames, self.mask_filenames):
                image = Image.open(os.path.join(self.image_dir, image_filename))
                mask = Image.open(os.path.join(self.mask_dir, mask_filename))

                if i == 0:
                    # Save the original image in the 1st iteration
                    transformed_image, transformed_mask = image, mask
                else:
                    transform = transforms.Compose([
                        transforms.RandomResizedCrop(size=(512, 512), scale=(1.2, 2)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip()
                    ])
                    transformed_image, transformed_mask = transform(image, mask)

                    transformed_image = transforms.ColorJitter(
                        brightness=(0.8, 1.3), hue=(-0.15, 0.15)
                    )(transformed_image)

                # Get the filename of the new image (name + version)
                transformed_image_filename = image_filename[:-4] + f'v{i}'
                transformed_mask_filename = mask_filename[:-4] + f'v{i}'

                # Get the full path of the image
                image_path = os.path.join(self.augmented_image_dir, transformed_image_filename)
                mask_path = os.path.join(self.augmented_mask_dir, transformed_mask_filename)

                transformed_image.save(image_path + '.jpg')
                transformed_mask.save(mask_path + '.png')

        self.augmented_image_filenames = sorted(os.listdir(self.augmented_image_dir))
        self.augmented_mask_filenames = sorted(os.listdir(self.augmented_mask_dir))


# Example of usage
if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(1.25, 1.25), hue=(-0.15, 0.15)),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Optional normalization
    ])

    dataset = SatelliteImageDataset(
        image_dir='final_more/images',
        mask_dir='final_more/masked',
        transform=transform  # Pass transformations to the dataset
    )

    img, mask = dataset[0]
    print(img.shape)
    print(mask.shape)

    mask = torch.tensor(class_index_to_image(mask.unsqueeze(0), dataset.color_map))
    print(mask.shape)

    # Visualize the image and mask
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("image.png")
    plt.show()

    # Save mask visualization (optional)
    plt.imshow(mask.squeeze(0).permute(1, 2, 0))
    plt.savefig("mask.png")
    plt.show()