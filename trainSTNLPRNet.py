import sys
sys.path.append("Automatic_Number_Plate_Recognition")

from model.STNLPRNet import build_stnlprnet
from decoderGreedy import Greedy_Decode_Eval
from trainModel import trainModel
from data.load_data import CHARS
from torch.utils.data import *
from torch import optim
import torch

class trainSTNLPRNet(trainModel):
    def __init__(self, args):
        super(trainSTNLPRNet, self).__init__(args)

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

    def saveFinalParameter(self):
        # save final parameters
        save_path = self.args.save_folder + 'Final_STNLPRNet_model.pth'
        torch.save(self.stnlprnet.state_dict(), save_path)
        lprnet_eval = build_stnlprnet(lpr_max_len=self.args.lpr_max_len, phase=self.args.phase_train, class_num=len(CHARS), dropout_rate=self.args.dropout_rate, batch_size=self.args.test_batch_size)
        lprnet_eval.to(self.device)
        lprnet_eval.load_state_dict(torch.load(save_path))

        # final test
        print("Final test Accuracy:")
        Greedy_Decode_Eval([lprnet_eval], self.test_dataset, self.args)

    def models(self):
        return [self.stnlprnet]
