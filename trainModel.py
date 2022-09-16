import os
import torch
from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
import torch.nn as nn
from abc import ABC, abstractmethod

class trainModel(ABC):
    def __init__(self, args):
        self.args = args

        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

        # build networks
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        train_img_dirs = os.path.expanduser(args.train_img_dirs)
        test_img_dirs = os.path.expanduser(args.test_img_dirs)
        self.train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len, True)
        self.test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

        self.epoch_size = len(self.train_dataset) // args.train_batch_size
        self.max_iter = args.max_epoch * self.epoch_size

        self.ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

        if args.resume_epoch > 0:
            self.start_iter = args.resume_epoch * self.epoch_size
        else:
            self.start_iter = 0
    
    def defaultLprInit(self, model):
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        model.backbone.apply(weights_init)
        model.container.apply(weights_init)

    @abstractmethod
    def models(self):
        pass

    def sparse_tuple_for_ctc(self, T_length, lengths):
        input_lengths = []
        target_lengths = []

        for ch in lengths:
            input_lengths.append(T_length)
            target_lengths.append(ch)

        return tuple(input_lengths), tuple(target_lengths)

    def adjust_learning_rate(self, optimizer, cur_epoch, base_lr, lr_schedule):
        """
        Sets the learning rate
        """
        
        lr = 0
        for i, e in enumerate(lr_schedule):
            if cur_epoch < e:
                lr = base_lr * (0.1 ** i)
                break
        if lr == 0:
            lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr