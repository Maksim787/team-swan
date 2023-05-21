import os
import PIL
from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class Expand:
    def __call__(self, input_tensor):
        shapes_new = list(input_tensor.size())
        mx = max(shapes_new)
        max_size = [3, mx, mx]
        for i in range(3):
            if max_size[i] > shapes_new[i]:
                shapes_new[i] = max_size[i] - input_tensor.shape[i]
                dop_torch = torch.ones(shapes_new)
                input_tensor = torch.cat((input_tensor, dop_torch), i)
        return input_tensor


class DefaultDataset(Dataset):

    def __init__(self, classes_paths: list[Path], is_test: bool):
        super().__init__()

        self.transform = T.Compose([T.ToTensor(), Expand(), T.Resize((128, 128))])
        self.labels = []
        self.images = []
        self.files = []

        if not is_test:
            for i in range(len(classes_paths)):
                files = sorted(os.listdir(classes_paths[i]))
                file_paths = [classes_paths[i] / file for file in files]
                self.labels += [i] * len(file_paths)
                self.files += file_paths
        else:
            self.labels = [-1]
            self.files = [classes_paths[0]]

    def __len__(self):
        return max(len(self.images), len(self.files))

    def __getitem__(self, item):
        filename = self.files[item]
        label = self.labels[item]
        image = self.transform(PIL.Image.open(filename).convert('RGB'))
        return image, label


def get_dataloader(class_paths: list[Path], batch_size=64, is_test: bool = True) -> DataLoader:
    return DataLoader(DefaultDataset(class_paths, is_test=is_test), batch_size=batch_size, shuffle=False)
