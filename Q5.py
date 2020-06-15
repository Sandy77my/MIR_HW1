from librosa import load
from librosa.feature import chroma_stft
from utils import mirex_evaluate, KS, inv_key_map
from scipy.stats import pearsonr
from scipy.signal import medfilt, fftconvolve
import numpy as np
import os
from tqdm import tqdm

data_dir = 'BPS_piano'
# ref_prefix = 'REF_key_'
predict_dir = 'predict_results/'

key_map = {v: k for k, v in inv_key_map.items()}

if __name__ == '__main__':
    file_names = [".".join(f.split(".")[:-1])for f in os.listdir(data_dir) if f[-4:] == '.wav']
    file_names.sort(key=float)

    d = 10
    g = 100
    w = 641
    meanFilt = np.ones(w) / w
    meanFilt2 = np.ones(w // 2 // d + 1) / (w // 2 // d + 1)
    overall_acc = []

    sym2num = np.vectorize(inv_key_map.get)
    num2sym = np.vectorize(key_map.get, otypes=[np.str])
    evaluateVec = np.vectorize(mirex_evaluate, otypes=[float])

    for f in file_names:
        label = np.loadtxt(os.path.join(
            data_dir, f + '.txt'), dtype='str')
        t = sym2num(label[:, 1])

        data, sr = load(os.path.join(data_dir, f + '.wav'), sr=None)
        hopSize = int(sr / d)
        windowSize = hopSize * 2

        chromaVec = chroma_stft(
            y=data, sr=sr, hop_length=hopSize, n_fft=windowSize, base_c=False)
        chromaVec = np.apply_along_axis(
            fftconvolve, 1, chromaVec, meanFilt, 'same')

        if chromaVec.shape[1] > len(label) * d:
            chromaVec = chromaVec[:, :len(label) * d]
        elif chromaVec.shape[1] < len(label) * d:
            chromaVec = np.column_stack(
                (chromaVec, np.zeros((12, len(label) * d - chromaVec.shape[1]))))

        chromaVec = chromaVec.reshape(12, len(label), d).mean(axis=2)
        chromaVec = np.log(1 + g * chromaVec)

        chromaVec = np.apply_along_axis(
            fftconvolve, 1, chromaVec, meanFilt2, 'same')

        prob = np.zeros((KS.shape[0], chromaVec.shape[1]))
        for n in range(chromaVec.shape[1]):
            prob[:, n] = np.apply_along_axis(
                pearsonr, 1, KS, chromaVec[:, n])[:, 0]

        y = np.argmax(prob, axis=0)
        y = medfilt(y, 9)
        acc = evaluateVec(y, t).tolist()

        print(f + '.wav', format(acc.count(1) / len(acc), '.3%'),format(np.mean(acc), '.3%'))
        overall_acc += acc

        y_inv = num2sym(y)
        np.savetxt(os.path.join(predict_dir, 'PRED_key_' + f + '.txt'), np.column_stack((np.arange(len(y)), y_inv)),delimiter='\t', fmt='%s')

    print("***** Q5-BPS-FH *****")
    print('OverallAcc:\nacc1:\t{:.3%}\n mirex acc:\t{:.3%}'.format(overall_acc.count(1) / len(overall_acc),np.mean(overall_acc)))