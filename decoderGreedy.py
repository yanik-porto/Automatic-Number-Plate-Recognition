import sys
sys.path.append("Automatic_Number_Plate_Recognition")

from torch.utils.data import DataLoader
import time
import numpy as np
from torch.autograd import Variable
from data.load_data import CHARS
import torch
from abc import ABC, abstractmethod

class DecoderGreedy(ABC):
    def __init__(self, args):
        self.args = args

    def collate_fn(self, batch):
        imgs = []
        labels = []
        lengths = []
        filenames = []
        for _, sample in enumerate(batch):
            img, label, length, filename = sample
            imgs.append(torch.from_numpy(img))
            labels.extend(label)
            lengths.append(length)
            filenames.append(filename)
        labels = np.asarray(labels).flatten().astype(int)

        return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths, filenames)
        
    def Greedy_Decode_Eval(self, Nets, datasets):
        args = self.args

        # TestNet = Net.eval()
        epoch_size = len(datasets) // args.test_batch_size
        batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=self.collate_fn))

        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        t_chars = 0
        T_c = 0
        T_f = 0
        res_chars = np.zeros(len(CHARS))
        t1 = time.time()
        for i in range(epoch_size):
            # load train data
            images, labels, lengths, _ = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start+length]
                targets.append(label)
                start += length
            targets = np.array([el.numpy() for el in targets], dtype=object)

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = images
            for i, Net in enumerate(Nets):
                if i > 0:
                    prebs = self.prepBetweenModels(prebs)
                prebs = Net(prebs)

            # greedy decode
            prebs = prebs.cpu().detach().numpy()
            preb_labels = list()
            for i in range(prebs.shape[0]):
                preb = prebs[i, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label: # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            for i, label in enumerate(preb_labels):
                t_chars+=len(targets[i])
                for j in range(len(label)):
                    if j>=len(targets[i]):
                        continue
                    if label[j] == targets[i][j]:
                        res_chars[label[j]]+=1
                        T_c+=1
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                fuzzy = 0
                for x in range(len(label)):
                    if targets[i][x]==label[x]:
                        fuzzy += 1
                if fuzzy/len(label) >= 0.75:
                    T_f += 1
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1

        divisor = 0.000001 if Tp + Tn_1 + Tn_2 == 0 else Tp + Tn_1 + Tn_2
        t_chars = 0.000001 if t_chars == 0 else t_chars
        Acc = Tp * 1.0 / divisor
        print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, divisor))
        print(f"[Info] 75%+ Accuracy: {T_f/divisor} [{T_f}/{divisor}]")
        t2 = time.time()
        print(f'[Info] Char Accuracy:{T_c/t_chars} [{T_c}/{t_chars}] ')
        # print('Per char: ')
        # for i in range(10):
        #     print(i,": ",res_chars[i]/T_c)
        # for i in range(10,len(CHARS)-1):
        #     print(chr(55+i),': ',res_chars[i]/T_c)
        print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

        return Acc

        @abstractmethod
        def prepBetweenModels(self, inputs):
            return inputs

def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # dim0=t, dim1=c
    maxT, maxC = mat.shape
    label = ''
    blankIdx = len(classes)
    lastMaxIdx = maxC  # init with invalid label

    for t in range(maxT):
        maxIdx = np.argmax(mat[t, :])
        if maxIdx != lastMaxIdx and maxIdx != blankIdx:
            label += classes[maxIdx]

        lastMaxIdx = maxIdx

    return label

def probsToLabel(probs):
    mat = probs.squeeze().transpose()
    classes = CHARS[:-1]
    return ctcBestPath(mat, classes)