import sys
sys.path.append("Automatic_Number_Plate_Recognition")

from model.LPRNet import build_lprnet
from model.STNet import STNet
from model.STNetSquare import STNetSquare
from trainModel import trainModel
from data.load_data import CHARS, tensorToImages, imagesToTensor
from torch.utils.data import *
from torch import optim
import torch
import cv2
import os

class trainSTNLPRNetAdaptiv(trainModel):
    def __init__(self, args, areSquareImages=False):
        imgSize = (48, 48) if areSquareImages else (94, 24)

        super(trainSTNLPRNetAdaptiv, self).__init__(args, areSquareImages, imgSize)
        
        if self.areSquareImages:
            self.stnet = STNetSquare(batch_size=args.train_batch_size, w=48, h=48)
        else:
            self.stnet = STNet(batch_size=args.train_batch_size, w=94, h=24)
        self.stnet.to(self.device)

        self.lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
        self.lprnet.to(self.device)

        # load pretrained model
        if args.pretrained_model:
            self.lprnet.load_state_dict(torch.load(args.pretrained_model))
            print("load pretrained model successful!")
        else:
            self.defaultLprInit(self.lprnet)
            print("initial net weights successful!")

        # define optimizer
        optimizer_params = [
            {'params': self.stnet.parameters(), 'weight_decay': args.weight_decay},
            {'params': self.lprnet.parameters(), 'weight_decay': args.weight_decay}
        ]
        self.optimizer = optim.Adam(optimizer_params, lr=args.learning_rate, betas = [0.9, 0.999], eps=1e-08, weight_decay=args.weight_decay)

    def saveFinalParameter(self):
        save_path_stnet = os.path.join(self.args.save_folder, + 'Final_' + self.stnet.__class__.__name__ + '_model.pth')
        torch.save(self.stnet.state_dict(), save_path_stnet)
        if self.areSquareImages:
            stnet_eval = STNetSquare(self.args.test_batch_size, self.args.img_size[0], self.args.img_size[1])
        else:
            stnet_eval = STNet(self.args.test_batch_size, self.args.img_size[0], self.args.img_size[1])
        stnet_eval.to(self.device)
        stnet_eval.load_state_dict(torch.load(save_path_stnet))

        save_path = os.path.join(self.args.save_folder, 'Final_' + self.lprnet.__class__.__name__ + '_model.pth')
        torch.save(self.lprnet.state_dict(), save_path)
        lprnet_eval = build_lprnet(lpr_max_len=self.args.lpr_max_len, phase=self.args.phase_train, class_num=len(CHARS), dropout_rate=self.args.dropout_rate)
        lprnet_eval.to(self.device)
        lprnet_eval.load_state_dict(torch.load(save_path))

        # final test
        print("Final test Accuracy:")
        self.Greedy_Decode_Eval([stnet_eval, lprnet_eval], self.test_dataset)

    def models(self):
        return [self.stnet, self.lprnet]

    def prepBetweenModels(self, inputs):
        if self.areSquareImages:
            imagesTrans = []
            images = tensorToImages(inputs)
            for img in images:
                imagesTrans.append(self.transformSquared(img))
            return imagesToTensor(imagesTrans, self.device, (94, 24))
        else:
            return inputs

    def transformSquared(self, img):
        width = self.args.img_size[0]
        height = self.args.img_size[1]
        twiceWidth = width * 2
        halfHeight = height // 2
        remain = 0 if twiceWidth <= 94 else twiceWidth % 94
        assert(halfHeight == 24)
        assert(twiceWidth - remain == 94)
        img = cv2.hconcat([img[:halfHeight,:-remain], img[halfHeight:height,:]])
        return img
