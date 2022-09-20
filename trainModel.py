from decoderGreedy import DecoderGreedy
from data.load_data import CHARS, LPRDataLoader
import numpy as np
import os
import time
import torch
import torch.nn as nn
from abc import abstractmethod
from data.load_data import CHARS
from torch.autograd import Variable
from torch.utils.data import *
from torch.utils.tensorboard import SummaryWriter

class trainModel(DecoderGreedy):
    def __init__(self, args, areSquareImages=False):
        super(trainModel, self).__init__(args)

        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

        # build networks
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.areSquareImages = areSquareImages
        self.args.img_size = (48, 48) if self.areSquareImages else self.args.img_size #Check if useful to provide img_size in args

        train_img_dirs = os.path.expanduser(args.train_img_dirs)
        test_img_dirs = os.path.expanduser(args.test_img_dirs)
        self.train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len, True)
        self.test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

        self.epoch_size = len(self.train_dataset) // args.train_batch_size
        if self.epoch_size == 0 : self.epoch_size = 1
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

    def train(self):
        args = self.args

        T_length = 18 # args.lpr_max_len
        epoch = 0 + args.resume_epoch
        loss_val = 0
        GLOBAL_LOSS = np.inf

        writer = SummaryWriter()

        for iteration in range(self.start_iter, self.max_iter):
            if iteration % self.epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(DataLoader(self.train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=self.collate_fn))
                loss_val = 0
                epoch += 1

            if iteration !=0 and iteration % args.save_interval == 0:
                for model in self.models():
                    torch.save(model.state_dict(), args.save_folder + model.__class__.__name__ + '__epoch_' + repr(epoch) + '_iteration_' + repr(iteration) + '.pth')

            if (iteration + 1) % args.test_interval == 0:
                for model in self.models():
                    model.eval()
                Acc = self.Greedy_Decode_Eval(self.models(), self.test_dataset)
                writer.add_scalar("Accuracy/eval", Acc, epoch)

            start_time = time.time()
            # load train data
            images, labels, lengths, _ = next(batch_iterator)

            # get ctc parameters
            input_lengths, target_lengths = self.sparse_tuple_for_ctc(T_length, lengths)

            # update lr
            lr = self.adjust_learning_rate(self.optimizer, epoch, args.learning_rate, args.lr_schedule)

            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            # forward
            logits = images
            for i, model in enumerate(self.models()):
                if i > 0:
                    logits = self.prepBetweenModels(logits)
                logits = model(logits)
            log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
            log_probs = log_probs.log_softmax(2).requires_grad_()

            # backprop
            self.optimizer.zero_grad()
            loss = self.ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                continue
            loss.backward()
            self.optimizer.step()
            loss_val += loss.item()
            end_time = time.time()
            if iteration % 20 == 0:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % self.epoch_size) + '/' + repr(self.epoch_size)
                    + '|| Total iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                    'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))

                writer.add_scalar("Loss/train", loss_val, epoch)


            if loss.item() < GLOBAL_LOSS:
                GLOBAL_LOSS = loss.item()
                #torch.save(lprnet.state_dict(), args.save_folder+f"Min_loss({round(loss.item(),4)})_(epoch{repr(epoch)})_model.pth")
        
        self.saveFinalParameter()

    @abstractmethod
    def saveFinalParameter(self):
        pass

    def prepBetweenModels(self, inputs):
        return inputs