import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class KITTI360Dataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, split, 'image_2')
        self.label_dir = os.path.join(root_dir, split, 'label_2')

        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))

        image = Image.open(img_name).convert('RGB')
        # Load your label data as needed, you might need to parse the label file

        if self.transform:
            image = self.transform(image)

        # Assuming label_data is a torch tensor containing your label information
        sample = {'image': image, 'label': label_data}

        return sample

# Example usage:
transform = None  # You can add transformations if needed
kitti_dataset = KITTI360Dataset(root_dir='/path/to/kitti360', transform=transform)

# Create DataLoader
batch_size = 4
dataloader = DataLoader(kitti_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    images = batch['image']
    labels = batch['label']
    # Do something with the batch