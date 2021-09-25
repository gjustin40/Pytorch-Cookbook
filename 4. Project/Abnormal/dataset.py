import glob

from torch.utils.data import Dataset
from PIL import Image


class myDataset(Dataset):
    
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.image_list = glob.glob(image_path + '/*')
        
    def __len__(self):
        
        return len(self.image_list)
    
    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        
        image = Image.open(file_name)
        
        if self.transform:
            image = self.transform(image)
            
        return image, image  

def collate_fn(batch):
    return tuple(zip(*batch))