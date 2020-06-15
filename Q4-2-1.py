# -*- coding:utf-8 -*-
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import utils  # self-defined utils.py file

DB = 'GTZAN'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB + '/wav/*/*.wav')
    # print(FILES)
else:
    FILES = glob(DB + '/wav/*.wav')
    # print(FILES)

GENRE = [g.replace('\\','/').split('/')[2] for g in glob(DB+'/wav/*')]
GENRE.remove('classical')
# print(GENRE)
n_fft = 100  # (ms)
hop_length = 25  # (ms)

# %% Q4
if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
    RMajor, RMinor = defaultdict(list), defaultdict(list)
    coffiListsMaj, coffiListsMin = list(), list()
else:
    label, pred = list(), list()
    RMajor, RMinor = list(), list()
chromagram = list()
gens = list()

idx =0
gamma = 1
while idx<4 and gamma<1001:
    print('gamma',gamma)
    for f in tqdm(FILES):
        f = f.replace('\\', '/')
        # print("file: ", f)
        content = utils.read_keyfile(f, '*.lerch.txt')
        if (int(content) < 0): continue  # skip saving if key not found
        if DB == 'GTZAN':
            gen = f.split('/')[2]
            label[gen].append(utils.LABEL[int(content)])
            gens.append(gen)
        else:
            label.append(utils.LABEL[content])
    
        sr, y = utils.read_wav(f)
    
        cxx = librosa.get_duration(y=y, sr=sr)
        cxx = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=y, sr=sr)))
        chromagram.append(cxx)  # store into list for further use
        chroma_vector = np.sum(cxx, axis=1)
    
        KS = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
              "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    
        chroma_vector = chroma_vector.tolist()
        for i in range(11):
            majorCoefficient = pearsonr(chroma_vector, KS['major'])
            minorCoefficient = pearsonr(chroma_vector, KS['minor'])
            coffiListsMaj.append(majorCoefficient[0])
            coffiListsMin.append(minorCoefficient[0])
            temp = chroma_vector[0]
            chroma_vector.pop(0)
            chroma_vector.append(temp)
    
        maxRMajor = np.where(coffiListsMaj == np.amax(coffiListsMaj))
        maxRMinor = np.where(coffiListsMin == np.amax(coffiListsMin))
        maxRMajor = maxRMajor[0][0]
        maxRMinor = maxRMinor[0][0]
    
        if coffiListsMaj[maxRMajor]> coffiListsMin[maxRMinor]:
            key_ind = (maxRMajor+3)%12
        else:
            key_ind = (maxRMinor+15)%24
    
        #print('key_ind: {}\n'.format(key_ind))
    
        modePred = utils.lerch_to_str(key_ind)
    
        if DB == 'GTZAN':
            pred[gen].append(modePred)
        else:
            pred.append('?')  # you may ignore this when starting with GTZAN dataset
    
        coffiListsMaj = []
        coffiListsMin = []
    
    print("***** Q4 *****")
    if DB == 'GTZAN':
        label_list, pred_list = list(), list()
        print("Genre    \taccuracy")
        for g in GENRE:
            correct = 0
            for acc_len in range(len(label[g])):
                #print('{}\t{}'.format(label[g][acc_len],pred[g][acc_len]))
                if label[g][acc_len] == pred[g][acc_len]:
                    correct += 1
            try:
                acc = correct / len(label[g])
            except ZeroDivisionError:
                acc = 0
            print("{:9s}\t{:.2%}".format(g, acc))
            label_list += label[g]
            pred_list += pred[g]
    else:
        label_list = label
        pred_list = pred
    
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
    idx+=1
    gamma*=10