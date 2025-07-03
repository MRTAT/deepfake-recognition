
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms.v2 import Resize, Compose, ToTensor

class Deepfake(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train: mode = "train"
        else: mode = "test"
        root = os.path.join(root, mode)  # Root + train or test (dataset_std/test)

        self.transform = transform

        self.image_paths = list()
        self.labels = list()

        self.categories = ['fake', 'real']
        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(root, category)  # dataset_std/test/fake
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)  # dataset_std/test/fake/img_1.jpg
                self.image_paths.append(file_path)
                self.labels.append(i)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # Take path
        image = Image.open(image_path).convert("RGB")  # Open image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

if __name__ == '__main__':
    root = "dataset_std"

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    dataset = Deepfake(root, train=True, transform=None)
    image, label = dataset.__getitem__(8888)
    image.show()


