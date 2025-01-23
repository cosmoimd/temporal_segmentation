import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from torchvision.models import resnet50

class VideoFrameDataset(Dataset):
    """Dataset for loading video frames from a directory based on frame names in a CSV file."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (str): Path to the CSV file with frame names.
            root_dir (str): Directory with all the frames.
        """
        self.frame_info = pd.read_csv(csv_file, usecols=[0])  # Assuming frame names are in the first column
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet50 preprocessing
        ])

    def __len__(self):
        return len(self.frame_info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame_info.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        return image

def get_video_loader(csv_file, root_dir, batch_size=32, num_workers=4):
    """Utility function to get a video data loader."""
    dataset = VideoFrameDataset(csv_file, root_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Example usage
if __name__ == '__main__':
    loader = get_video_loader('path_to_csv/001-001.csv', 'path_to_frames/')
    for batch in loader:
        # batch is a tensor of shape (batch_size, 3, 224, 224)
        # Process the batch through ResNet50 or any other operations
        pass
