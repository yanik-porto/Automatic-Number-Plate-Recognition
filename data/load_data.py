from torch.utils.data import Dataset
import cv2
import albumentations as A
# import pandas as pd
import numpy as np
import random
import os
import sys
import torch
from torch.autograd import Variable
sys.path.append("Automatic_Number_Plate_Recognition")
from misc.separator import *

CHARS = [
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z','-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

def lpidToLabel(lpid):
    label = list()
    for c in lpid:
        c = c.upper()
        if not c in CHARS_DICT:
            print(lpid)
        label.append(CHARS_DICT[c])
    return label

def tensorToImages(tensor):
    images = []
    for entry in tensor:
        image = entry.cpu().detach().numpy()
        image /= .0078125
        image += 127.5
        image = np.transpose(image, (1, 2, 0))
        images.append(image)
    return images


def imagesToTensor(images, device, imgSize):
    tensors = []
    for img in images:
        if img.shape[0] != imgSize[1] or img.shape[1] != imgSize[0]:
            img = cv2.resize(img, imgSize)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        tensors.append(torch.from_numpy(img))
    tensors = torch.stack(tensors, 0)
    tensors = tensors.to(device)
    return Variable(tensors)

def transformSquared(img):
    halfHeight = img.shape[0] // 2
    img = cv2.hconcat([img[:halfHeight,:], img[halfHeight:halfHeight*2,:]]) # halfHeight * 2 instead of lastrow, because heights not equal for odd number of rows
    # quartWidth = img.shape[1] // 4
    # img = cv2.hconcat([img[:halfHeight,quartWidth:-quartWidth], img[halfHeight:halfHeight*2,:]])
    return img

def isSquareImg(img):
    return img.shape[0] * 2 > img.shape[1] 