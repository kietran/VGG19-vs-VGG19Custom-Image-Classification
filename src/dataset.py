from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np

class MotorbikeDataset(Dataset):
    def __init__(self, root_path, transform, is_train=True, img_size=224):
        if is_train:
            data_path = os.path.join(root_path, 'train')
        else:
            data_path = os.path.join(root_path, 'test')

        self.categories = []
        for sub_dir in os.listdir(data_path):
            self.categories.append(sub_dir)
        
        self.image_paths = []
        self.labels = []
        for sub_dir in os.listdir(data_path):
            sub_dir_path = os.path.join(data_path, sub_dir)
            for file in os.listdir(sub_dir_path):
                self.image_paths.append(os.path.join(sub_dir_path, file))
                self.labels.append(self.categories.index(sub_dir))

        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        try:
            label = self.labels[index]
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert("RGB")

            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (self.img_size, self.img_size))
            # image = np.transpose(image, (2, 0, 1))/255.0
            # image = torch.from_numpy(image).float()
        except (UnidentifiedImageError, IOError) as e:
            # Return a blank or placeholder image if original is unreadable
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        return image, label