import os
import sys
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


ROOT_DIR = None


def setup():
    global ROOT_DIR
    ROOT_DIR = os.path.dirname(sys.modules['__main__'].__file__)

def load_from_file(link):
    # print(ROOT_DIR + link)
    entries = os.listdir(ROOT_DIR + link)
    images = []
    # sss = 0
    for entry in entries:
        # print(ROOT_DIR + link + '/' + entry)
        img = (Image.open(ROOT_DIR + link + '/' + entry).convert('RGB'))
        images += [img]
        # sss += 1
        # if(sss > 100):
        #     return images
    return images
def load_data():
    setup()
    dataset = []
    dataset.append([])
    dataset.append([])
    classes = 0
    for i in ['/data/разметка_кликун/klikun/images', '/data/разметка_малый/images/team-swan/data/разметка_шипун/images', '/data/разметка_малый/images']:
        goose1 = load_from_file(i)
        dataset[0].append(goose1)
        dataset[1].append([classes]*len(goose1))
        classes += 1
load_data()
