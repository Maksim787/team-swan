import os
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


class SwanDataset(Dataset):
    SPLIT_RANDOM_SEED = 1
    TEST_SIZE = 0.25
    PATHES = ['klikun/images', 'разметка_шипун/images', 'разметка_малый/images']
    LABELS = [0, 1, 2]

    def __init__(self, data_folder: str, load_to_ram=False, train=True, transform=None):
        super().__init__()

        # dataloader parameters
        self.data_folder = Path(data_folder)
        assert self.data_folder.exists() and self.data_folder.name == 'data'
        self.load_to_ram = load_to_ram
        self.train = train
        self.transform = transform

        # images and labels
        self.labels = []
        self.images = []
        self.files = []

        # iterate over labels and fill images and labels
        for path, label in zip(self.PATHES, self.LABELS):
            new_files_all = sorted(os.listdir(self.data_folder / path))
            # split each class uniformly with TEST_SIZE
            new_train_files, new_test_files = train_test_split(new_files_all, random_state=self.SPLIT_RANDOM_SEED + label, test_size=self.TEST_SIZE, shuffle=True)
            new_files_dataset = new_train_files if self.train else new_test_files
            new_files_dataset = [Path(path) / file for file in new_files_dataset]
            self.labels += [label] * len(new_files_dataset)
            if self.load_to_ram:
                # load images
                self.images += self._load_images(self.data_folder, new_files_dataset)
            else:
                # save filenames to load images later
                self.files += new_files_dataset

    def _load_images(self, path: Path, files: list[str]):
        images = []
        for filename in tqdm(files):
            # load image
            image = Image.open(path / filename).convert('RGB')
            images += [image]
        return images

    def __len__(self):
        return max(len(self.images), len(self.files))

    def __getitem__(self, item):
        if self.load_to_ram:
            # get image from memory
            image = self.images[item]
        else:
            # load image
            filename = self.files[item]
            image = Image.open(self.data_folder / filename).convert('RGB')
        label = self.labels[item]

        # apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_folder: str, batch_size=64, load_to_ram=False, train_transform=None, test_transform=None) -> tuple[DataLoader, DataLoader]:
    # load datasets
    train_set = SwanDataset(data_folder, train=True, load_to_ram=load_to_ram, transform=train_transform)
    test_set = SwanDataset(data_folder, train=False, load_to_ram=load_to_ram, transform=test_transform)

    # make dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # return dataloaders
    return train_loader, val_loader
