import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import os
from PIL import Image
import tifffile as tiff

# 自动检测设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, split='train'):
        super(NPY_datasets, self).__init__()

        images_list = os.listdir(os.path.join(path_Data, 'images'))
        masks_list = os.listdir(os.path.join(path_Data, 'masks'))

        # Load image and mask paths
        self.data = []
        for i in range(len(images_list)):
            img_path = os.path.join(path_Data, 'images', images_list[i])
            mask_path = os.path.join(path_Data, 'masks', masks_list[i])
            self.data.append([img_path, mask_path])

        # Shuffle the data
        np.random.shuffle(self.data)

        # Split the data into training, validation, and test sets
        total_len = len(self.data)
        train_len = int(total_len * 0.8)
        val_len = int(total_len * 0.1)
        test_len = total_len - train_len - val_len

        # Perform the split based on the `split` argument
        train_data, val_data, test_data = random_split(self.data, [train_len, val_len, test_len])

        if split == 'train':
            self.data = [self.data[idx] for idx in train_data.indices]
            self.transformer = config.train_transformer
        elif split == 'val':
            self.data = [self.data[idx] for idx in val_data.indices]
            self.transformer = config.test_transformer  # 使用测试集转换器
        elif split == 'test':
            self.data = [self.data[idx] for idx in test_data.indices]
            self.transformer = config.test_transformer
        else:
            raise ValueError("split must be 'train', 'val', and 'test'")

    def load_image(self, img_path):
        try:
            if img_path.endswith('.tif'):
                # Use tifffile to read TIFF images
                img = tiff.imread(img_path)
                if img.ndim == 2:
                    # If grayscale, repeat channels to create RGB
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] > 3:
                    # If more than 3 channels, keep only the first 3
                    img = img[:, :, :3]
            else:
                # Use PIL for other image formats
                img = np.array(Image.open(img_path).convert('RGB'))
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def load_mask(self, mask_path):
        try:
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = np.expand_dims(mask, axis=2) / 255  # Normalize to [0, 1]
            return mask
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]

        # Load image and mask
        img = self.load_image(img_path)
        if img is None:
            raise ValueError(f"Unable to load image at {img_path}")

        msk = self.load_mask(msk_path)
        if msk is None:
            raise ValueError(f"Unable to load mask at {msk_path}")

        # Apply transformations
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
