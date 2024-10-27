import builtins
import os
from config import config
import boto3
from smart_open import open
import albumentations as A
import cv2
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, bucket_name=str, prefix=str, transforms=None):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.prefix = prefix  # S3 folder prefix like 'data/train/'
        self.transforms = transforms
        self.data = []
        self.class_map = {}
        self.extensions = ("jpeg", "jpg", "png")
        
        # Get list of class folders and images within each class folder from S3
        self._load_s3_data()

    def _load_s3_data(self):
        """Fetches all image paths and class labels from the S3 bucket."""
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix, Delimiter='/')
        
        # Iterate through each folder (class) in the prefix
        if 'CommonPrefixes' in response:
            for class_folder in response['CommonPrefixes']:
                class_name = class_folder['Prefix'].split('/')[-2]  # Get the class name
                self.class_map[class_name] = len(self.class_map)  # Assign a unique index
                
                # Get all images within the class folder
                class_response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=class_folder['Prefix'])
                for item in class_response.get('Contents', []):
                    file_ext = item['Key'].split('.')[-1].lower()
                    if file_ext in self.extensions:
                        self.data.append([item['Key'], class_name])  # Store S3 path and class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_key, class_name = self.data[idx]
        
        # Load the image directly from S3
        with open(f's3://{self.bucket_name}/{img_key}', 'rb') as file:
            image = Image.open(file).convert("RGB")
            image = np.array(image)  # Convert PIL image to numpy array for albumentations

        # Applying transforms on the image
        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = self.class_map[class_name]
        return image, label

def get_transform():
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    return A.Compose([resize, normalize, to_tensor])

if __name__ == "__main__":
    # Initialize dataset with S3 bucket details
    bucket_name = config.S3_DATA_BUCKET
    prefix = config.S3_DATA_PREFIX + "train/"
    print(bucket_name, "\n", prefix)
    
    dataset = CustomDataset(bucket_name=bucket_name, prefix=prefix, transforms=get_transform())
    
    # DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    total_imgs = 0
    for imgs, labels in data_loader:
        total_imgs += int(imgs.shape[0])
        print(imgs.shape)
        break
    print(total_imgs)

