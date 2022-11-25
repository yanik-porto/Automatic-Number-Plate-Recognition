from torch.utils.data import Dataset
import cv2
import albumentations as A
import numpy as np
import random
import os
import sys
from torch.autograd import Variable
sys.path.append("Automatic_Number_Plate_Recognition")
from misc.separator import *
from data.load_data import lpidToLabel, transformSquared
from data.dbReader import readAnnotAsRectAndGt

class PlateDataset(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, augment=False, suffix='__ctx.jpg'):
        self.img_dir = img_dir
        self.imgWithAnnots = []
        self.augment = augment
        for i in range(len(img_dir)):
            print("Peak files in ", img_dir[i])
            for root, _, files in os.walk(img_dir[i]):
                for f in files:
                    if f.endswith(tuple(suffix.split(','))):
                        imgPath = os.path.join(root, f)
                        annotPath = os.path.join(root, f[:-len(suffix)] + '__rct.txt') # TODO : handle multiple suffixes
                        if (os.path.isfile(annotPath)):
                            rectAngGts = readAnnotAsRectAndGt(annotPath)
                            if len(rectAngGts) > 0:
                                for rectgt in rectAngGts:
                                    self.imgWithAnnots.append((imgPath, rectgt))

        print(len(img_dir), " dir found, size: ",len(self.imgWithAnnots))
        random.shuffle(self.imgWithAnnots)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.doTransformSquare = False

    def __len__(self):
        return len(self.imgWithAnnots)

    def __getitem__(self, index):
        imgPath, rectAndGt = self.imgWithAnnots[index]
        ctxImg = cv2.imread(imgPath)
        if ctxImg is None:
            print(ctxImg)

        plateImg = self.cropCtxWithAugmentation(ctxImg, rectAndGt[0])

        if self.doTransformSquare:
            plateImg = transformSquared(plateImg)
        plateImg = cv2.resize(plateImg, self.img_size)
            
        plateImg = self.transform(plateImg)

        lpid = rectAndGt[1]
        label = lpidToLabel(lpid)
        label_length = len(label)

        return plateImg, label, label_length, ctxImg

    def transform(self, img):
        if self.augment:
            img = self.augment_image(img)
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
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

    def cropCtxWithAugmentation(self, ctxImg, rect):        
        # only roi for now
        arect = rect

        # crop with augmented rectangle
        roi = ctxImg[arect[1]:arect[3], arect[0]:arect[2]]
        return roi