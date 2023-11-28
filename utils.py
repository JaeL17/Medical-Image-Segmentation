import os
import cv2
import requests
import io
from glob import glob
import zipfile
from dataclasses import dataclass
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES= 4 # 3 + 1 including background
    IMAGE_SIZE= (288, 288) # W, H
    MEAN=(0.485, 0.456, 0.406)
    STD=(0.229, 0.224, 0.225)
    BACKGROUND_CLS_I = 0
    URL = r"https://www.dropbox.com/scl/fi/r0685arupp33sy31qhros/dataset_UWM_GI_Tract_train_valid.zip?rlkey=w4ga9ysfiuz8vqbbywk0rdnjw&dl=1"
    DATASET_PATH= os.path.join(os.getcwd(), "dataset_UWM_GI_Tract_train_valid")

@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES= os.path.join(DatasetConfig.DATASET_PATH, "train", "images", r"*.png")
    DATA_TRAIN_LABELS = os.path.join(DatasetConfig.DATASET_PATH, "train", "masks",  r"*.png")
    DATA_VALID_IMAGES= os.path.join(DatasetConfig.DATASET_PATH, "valid", "images", r"*.png")
    DATA_VALID_LABELS = os.path.join(DatasetConfig.DATASET_PATH, "valid", "masks",  r"*.png")

@dataclass
class InferenceConfig:
    BATCH_SIZE= 10
    NUM_BATCHES = 2
     
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_classes=10,
        img_size=(384, 384),
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle_validation=False,
    ):
        super().__init__()
 
        self.num_classes = num_classes
        self.img_size = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_validation = shuffle_validation
 
    def prepare_data(self):

        dataset_zip_path = f"{DatasetConfig.DATASET_PATH}.zip"
 
        # Download if dataset does not exists.
        if not os.path.exists(DatasetConfig.DATASET_PATH):
 
            print("Downloading dataset.", end="")
            file = requests.get(DatasetConfig.URL)
            open(dataset_zip_path, "wb").write(file.content)
 
            try:
                with zipfile.ZipFile(dataset_zip_path) as z:
                    z.extractall(os.path.split(dataset_zip_path)[0]) # Unzip where downloaded.
                print("Done")
            except:
                print("Invalid file")
 
            os.remove(dataset_zip_path) # Remove the ZIP file to free storage space.
 
    def setup(self, *args, **kwargs):
        # Create training dataset and dataloader.
        train_imgs = sorted(glob(f"{Paths.DATA_TRAIN_IMAGES}"))
        train_msks  = sorted(glob(f"{Paths.DATA_TRAIN_LABELS}"))
 
        # Create validation dataset and dataloader.
        valid_imgs = sorted(glob(f"{Paths.DATA_VALID_IMAGES}"))
        valid_msks = sorted(glob(f"{Paths.DATA_VALID_LABELS}"))
        self.train_ds = BuildDataset(image_paths=train_imgs, mask_paths=train_msks, img_size=self.img_size,  
                                       is_train=True, ds_mean=self.ds_mean, ds_std=self.ds_std)

        self.valid_ds = BuildDataset(image_paths=valid_imgs, mask_paths=valid_msks, img_size=self.img_size, 
                                       is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
    def train_dataloader(self):
        # Create train dataloader object with drop_last flag set to True.
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )    
 
    def val_dataloader(self):
        # Create validation dataloader object.
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        ) 
    
class BuildDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths  
        self.is_train = is_train
        self.img_size = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
 
    def __len__(self):
        return len(self.image_paths)
 
    def setup_transforms(self, *, mean, std):
        transforms = []
 
        # Augmentation for training set
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
            ])
 
        # Preprocess transforms - Normalisation and converting to PyTorch tensor format (HWC --> CHW)
        transforms.extend([
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(always_apply=True),  # (H, W, C) --> (C, H, W)
        ])
        return A.Compose(transforms)
 
    def load_file(self, file_path, strr, depth=0):
        file = cv2.imread(file_path, depth) # load file
        if depth == cv2.IMREAD_COLOR:
            file = file[:, :, ::-1]

        return cv2.resize(file, (self.img_size), interpolation=cv2.INTER_NEAREST) # resizing image to pre-defined size
 
    def __getitem__(self, index):
        # Load image and mask file
        image = self.load_file(self.image_paths[index],'img', depth=cv2.IMREAD_COLOR)
        mask = self.load_file(self.mask_paths[index], 'mask', depth=cv2.IMREAD_GRAYSCALE)

        # Apply Preprocessing (+ Augmentations) transformations to image-mask pair
        transformed = self.transforms(image=image, mask=mask)
             
        image, mask = transformed["image"], transformed["mask"].to(torch.long)
        return image, mask
    
def denormalize(tensors, *, mean, std):
    for c in range(3):#CHANNELS):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
 
    return torch.clamp(tensors, min=0.0, max=1.0)

# Create a mapping of class ID to RGB value
id2color = {
    0: (0, 0, 0),    # background pixel
    1: (0, 0, 255),  # Blue - Stomach
    2: (0, 255, 0),  # Green - Small Bowel
    3: (255, 0, 0),  # Red - large Bowel
}

def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
 
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]

    return np.float32(output) / 255.0

def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
 
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    return np.clip(image, 0.0, 1.0)