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

class LPRDataset(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, augment=False, suffix='__plaque.jpg'):
        self.img_dir = img_dir
        self.img_paths = []
        self.augment = augment
        for i in range(len(img_dir)):
            print("Peak files in ", img_dir[i])
            for root, _, files in os.walk(img_dir[i]):
                for f in files:
                    if f.endswith(tuple(suffix.split(','))):
                        self.img_paths.append(os.path.join(root, f))
            # self.img_paths += [el for el in paths.list_images(img_dir[i])]

        print(len(img_dir), " dir found, size: ",len(self.img_paths))
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.doTransformSquare = False

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        if Image is None:
            print(filename)
        height, width, _ = Image.shape

        if self.doTransformSquare:
            Image = transformSquared(Image)
        Image = cv2.resize(Image, self.img_size)
            
        # if width/height<2:
        #     Image = bifurcate(Image)
        Image = self.transform(Image)

        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        label = self.labelFromImgname(imgname)
        label_length = len(label)

        # if label_length<8 and index!=len(self.img_paths)-1:
        #     Image, label, label_length, filename = self.__getitem__(index+1)
        return Image, label, label_length, filename

    def labelFromImgname(self, imgname):
        """
        Return the label of a given image name (by convention : UID__LPID__IMGTYPE)
        if image name not in 3 parts, then return empty list
        """
        label = list()

        parts = imgname.split("__")
        if len(parts) < 3:
            print("wrong img name")
            return label

        lpid = parts[1]
        for c in lpid:
            c = c.upper()
            label.append(CHARS_DICT[c])

        return label

    def transform(self, img):
        if self.augment:
            img = self.augment_image(img)
        img = img.astype('float32')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img -= 127.5
        img *= 0.0078125
        #thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        #img = np.reshape(img, img.shape + (1,))
        img = np.transpose(img, (2, 0, 1))
        return img

    def augment_image(self, image):
        transform = A.Compose([
        A.GaussNoise(),
        A.OneOf([
            A.MotionBlur(p=.4),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Affine(rotate=0.5, shear=0.5, p=0.3)
        ])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        return augmented_image