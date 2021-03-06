#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import utils  # self-defined utils.py file

DB = 'Giantsteps'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB + '/wav/*/*.wav')
else:
    FILES = glob(DB + '/wav/*.wav')
# print(FILES)

n_fft = 100  # (ms)
hop_length = 25  # (ms)

if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()

index =0
gamma = 1
while index<4 and gamma<1001:
    print('gamma',gamma)
    for f in tqdm(FILES):
        f = f.replace('\\', '/')
        # print("file: ", f)
        content = utils.read_keyfile(f, '*.key')
        if (len(content) < 0): continue  # skip saving if key not found
        if DB == 'GTZAN':
            gen = f.split('/')[2]
            label[gen].append(utils.LABEL[int(content)])
            gens.append(gen)
        else:
            label.append(content)

        sr, y = utils.read_wav(f)

        # gamma = input("gamma (1, 10, 100, 1000): ")
        cxx = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=y, sr=sr)))
        chromagram.append(cxx)  # store into list for further use
        chroma_vector = np.sum(cxx, axis=1)
        key_ind = np.where(chroma_vector == np.amax(chroma_vector))
        key_ind = int(key_ind[0])

        MODE = {"major": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                "minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]}
        MODE['major'] = utils.rotate(MODE['major'], key_ind)
        MODE['minor'] = utils.rotate(MODE['minor'], key_ind)
        r_co_major = pearsonr(chroma_vector, MODE["major"])
        r_co_minor = pearsonr(chroma_vector, MODE["minor"])
        # print(r_co_major[0])
        # print(r_co_minor[0])

        mode = ''
        if DB == 'Giantsteps':
            Cmajor_annoatation = (key_ind+3)%12
            if (r_co_major[0] > r_co_minor[0]):
                mode = Cmajor_annoatation
            else:
                mode = Cmajor_annoatation+12
            mode = utils.lerch_to_str(mode)
        else:
            if (r_co_major[0] > r_co_minor[0]):
                mode = key_ind
            else:
                mode = key_ind+12
            mode = utils.lerch_to_str(mode)
            # print('mode', mode)

        if DB == 'Giantsteps':
            pred.append(mode)
        else:
            pred.append('?')  # you may ignore this when starting with GTZAN dataset
        # print(pred[gen])

        label_list = label
        pred_list = pred

        for idx, item in enumerate(label_list):
            a = ' '.join(item.split()[0]).split()
            if (len(a)>1):
                temp = item.split()[0] = a[0]+'#'
                label_list[idx] = temp + ' ' + item.split()[1]

    print("***** Q2 *****")
    if DB == 'Giantsteps':
        correct_all = 0
        for acc_len in range(len(label_list)):
            if label_list[acc_len] == pred_list[acc_len]:
                correct_all += 1
        try:
            acc_all = correct_all / len(label_list)
        except ZeroDivisionError:
            acc_all = 0
    print("----------")
    print("Overall accuracy:\t{:.2%}".format(acc_all))
    index+=1
    gamma*=10
