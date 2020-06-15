from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, Flatten, Activation
from keras.optimizers import Adam
from librosa.feature import chroma_stft
from librosa import load
from scipy.signal import fftconvolve
import numpy as np
import os
import matplotlib.pyplot as plt
from pretty_midi import key_number_to_key_name
from utils import mirex_evaluate, KS, inv_key_map

import utils

data_dim = 12
timesteps = 12
num_classes = 24

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=(12,12,1)))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(24))
model.add(Activation('softmax'))

Adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()


data_dir = 'C:/Users/Sandy/Desktop/MIR/HW1/BPS_piano'
# ref_prefix = 'REF_key_'

sym2num = np.vectorize(inv_key_map.get)
evaluate_vec = np.vectorize(mirex_evaluate, otypes=[float])

train_set = [1, 3, 5, 11, 16, 19, 20, 22, 25, 26, 32]
val_set = [6, 13, 14, 21, 23, 31]
test_set = [8, 12, 18, 24, 27, 28]

train_data_list = []
train_target_list = []
val_data_list = []
val_target_list = []
test_data_list = []
test_target_list = []

w = 769
mean_filt = np.ones(w) / w

file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(data_dir) if f[-4:] == '.wav']

for f in file_names:
    label = np.loadtxt(os.path.join(data_dir, f + '.txt'), dtype='str')
    t = sym2num(label[:, 1])
    t2 = np.zeros((len(t), num_classes))
    t2[(np.arange(len(t)), t)] = 1

    data, sr = load(os.path.join(data_dir, f + '.wav'), sr=None)
    hop_size = int(sr / timesteps)
    window_size = hop_size * 2

    chroma_a = chroma_stft(y=data, sr=sr, hop_length=hop_size, n_fft=window_size, base_c=False)
    chroma_a = np.apply_along_axis(fftconvolve, 1, chroma_a, mean_filt, 'same')
    
    if chroma_a.shape[1] > len(t) * timesteps:
        chroma_a = chroma_a[:, :len(t) * timesteps]
    elif chroma_a.shape[1] < len(t) * timesteps:
        chroma_a = np.column_stack((chroma_a, np.zeros((data_dim, len(t) * timesteps - chroma_a.shape[1]))))
    chroma_a = chroma_a.T.reshape(-1, timesteps, data_dim)
    
    if int(f) in train_set:
        train_data_list.append(chroma_a)
        train_target_list.append(t2)
    elif int(f) in val_set:
        val_data_list.append(chroma_a)
        val_target_list.append(t2)
    else:
        test_data_list.append(chroma_a)
        test_target_list.append(t2)
    
    # print(f, chroma_a.shape, t2.shape)
# print(len(train_data_list), len(val_data_list), len(test_data_list))

train_data = np.concatenate(train_data_list)
train_target = np.concatenate(train_target_list)
val_data = np.concatenate(val_data_list)
val_target = np.concatenate(val_target_list)
test_data = np.concatenate(test_data_list)
test_target = np.concatenate(test_target_list)

train_data = np.expand_dims(train_data, axis=3)
val_data = np.expand_dims(val_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)
# print(train_data.shape)


history = model.fit(train_data, train_target, batch_size=100, epochs=40, validation_data=(val_data, val_target))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(test_data, test_target)

test_y = model.predict(test_data)
test_y = np.argmax(test_y, axis=1)
test_t = np.argmax(test_target, axis=1)
print(test_t.shape, test_y.shape)
acc = evaluate_vec(test_y, test_t).tolist()

print(format(acc.count(1) / len(acc), '.6%'), format(np.mean(acc), '.6%'))