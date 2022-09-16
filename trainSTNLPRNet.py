import sys
sys.path.append("Automatic_Number_Plate_Recognition")

from model.StnLprNet import build_stnlprnet
from decoderGreedy import Greedy_Decode_Eval, collate_fn
from trainModel import trainModel
from data.load_data import CHARS
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter

class trainSTNLPRNet(trainModel):
    def __init__(self, args):
        super(trainSTNLPRNet, self).__init__()

        self.stnlprnet = build_stnlprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate, batch_size=args.train_batch_size)
        self.stnlprnet.to(self.device)

        # load pretrained model
        if args.pretrained_model:
            self.stnlprnet.load_state_dict(torch.load(args.pretrained_model))
            print("load pretrained model successful!")
        else:
            self.defaultLprInit(self.stnlprnet)
            print("initial net weights successful!")

        # define optimizer
        self.optimizer = optim.Adam(self.stnlprnet.parameters(), lr=args.learning_rate, betas = [0.9, 0.999], eps=1e-08,
                            weight_decay=args.weight_decay)

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
                batch_iterator = iter(DataLoader(self.train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
                loss_val = 0
                epoch += 1

            if iteration !=0 and iteration % args.save_interval == 0:
                torch.save(self.stnlprnet.state_dict(), args.save_folder + 'STNLPRNet_' + '_epoch_' + repr(epoch) + '_iteration_' + repr(iteration) + '.pth')

            if (iteration + 1) % args.test_interval == 0:
                testnet = self.stnlprnet.eval()
                Acc = Greedy_Decode_Eval(testnet, self.test_dataset, args)
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
            logits = self.stnlprnet(images)
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

    def saveFinalParameter(self):
        # save final parameters
        save_path = self.args.save_folder + 'Final_STNLPRNet_model.pth'
        torch.save(self.stnlprnet.state_dict(), save_path)
        lprnet_eval = build_stnlprnet(lpr_max_len=self.args.lpr_max_len, phase=self.args.phase_train, class_num=len(CHARS), dropout_rate=self.args.dropout_rate, batch_size=self.args.test_batch_size)
        lprnet_eval.to(self.device)
        lprnet_eval.load_state_dict(torch.load(save_path))

        # final test
        print("Final test Accuracy:")
        Greedy_Decode_Eval(lprnet_eval, self.test_dataset, self.args)