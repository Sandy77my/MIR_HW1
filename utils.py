#!/usr/bin/python
# -*- coding:utf-8 -*-

from scipy.io import wavfile as wav
import numpy as np
from pretty_midi import key_number_to_key_name

def rotate(ar,n):
	"""Rotate right an list/array by n items.

	Args:
		ar: Input array to be rotated.
		n: Number of items to be rotated.
	"""
	return ar[-n%len(ar):]+ar[:-n%len(ar)]

MODE = {"major":[1,0,1,0,1,1,0,1,0,1,0,1],
        "minor":[1,0,1,1,0,1,0,1,1,0,1,0]}
KS   = {"major":[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
        "minor":[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]}
KEY  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
# According to alexander lerch's encoding
# From A to G# and from major to minor
LABEL = [k+" major" if i<12 else k+" minor" for i,k in enumerate(rotate(KEY,3)*2)]
LABEL_str = dict(zip(LABEL,list(range(len(LABEL)))))

DATA_SPLIT= {'BPS-FH': {'train':[1, 3, 5, 11, 16, 19, 20, 22, 25, 26, 32],
						'valid':[6, 13, 14, 21, 23, 31],
						'test' :[8, 12, 18, 24, 27, 28]},
			 'A-MAPS': {'train':'abcdefghijkl',
						'valid':'mnop',
						'test' :'stuvwxyz'}}

def read_wav(f):
	"""Read wav audio and reformat type.

	Read in wav file and reformat the data type to 32-bit floating-point. And 
	then, flatten to mono if it was stereo.

	Args:
		f: The audio filename.
	Returns:
		sr: Sampling rate of wav file.
		y: Data read from wav file.
	"""
	sr,y = wav.read(f)
	
	if y.dtype==np.int16: y = y/2**(16-1)
	elif y.dtype==np.int32: y = y/2**(32-1)
	elif y.dtype==np.int8: y = (y-2**(8-1))/2**(8-1)
	
	if y.ndim==2:
		y = y.mean(axis=1)
	return (sr,y)

def read_keyfile(f,keyname='*.txt'):
	f = f.replace('\\','/')
	prefix, suffix = keyname.split('*')
	return open(f.replace('/wav/','/key/'+prefix).replace('.wav',suffix),'r').read().strip()
def parse_key(ar):
	"""Parse key name of BPS-FH dataset.
	"""
	MAJOR = 'CDEFGAB'
	ar = [v.upper()+' major' if v[0] in MAJOR else v.upper()+' minor' for v in ar]
	ar = [MAJOR[MAJOR.index(v[0])-1]+v[1:].replace('-','#') if '-' in v else v for v in ar]
	ar = [v.replace('B#','B').replace('E#','E') for v in ar]
	ar = [v.replace('+','#') if '+' in v else v for v in ar]
	return ar

def generalize_key(key):
	"""Parse general key name to mir_eval format.
	
	For generality, we use key name according to the form '(key) (mode)'. Though
	certain key strings are equivalent, e.g. 'C# major' and 'Db major', this 
	function transform all into only '#' exist.
	
	"""
	MAJOR = 'CDEFGAB'
	MINOR = 'cdefgab'
	key = key.lower()
	if key[1]=='b':
		key = MAJOR[MINOR.index(key[0])-1] + '#' + key[2:]
	else:
		key = key[0].upper() + key[1:]
	key = key.replace('B#','B').replace('E#','E')
	return key

def lerch_to_str(lerch): return LABEL[lerch] # get key_mode str by lerch num
def str_to_lerch(key): return LABEL_str[key] # get lerch num by key_mode str

def parse_key_number(key_number):
	"""Parse key_number to string in pretty_midi.KeySignature.
	"""
	key = key_number_to_key_name(key_number)
	return generalize_key(key)

def mirex_evaluate(y, t):
    y_is_minor = y // 12
    t_is_minor = t // 12
    if y == t:
        return 1
    elif t_is_minor == y_is_minor and (t + 7) % 12 == y % 12:  # perfect fifth
        return 0.5
    elif y_is_minor > t_is_minor and (y + 3) % 12 == t % 12:  # relative minor
        return 0.3
    elif y_is_minor < t_is_minor and (t + 3) % 12 == y % 12:  # relative major
        return 0.3
    elif t_is_minor != y_is_minor and t % 12 == y % 12:  # parallel
        return 0.2
    else:
        # print("Wrong: ", key_map[t], "is not", key_map[y])
        return 0

key = ['A', 'B-', 'B', 'C', 'D-', 'D', 'E-', 'E', 'F', 'G-', 'G', 'A-']
tmp = [k.lower() for k in key]
key += tmp
inv_key_map = dict(zip(key, range(24)))

key2 = ['A', 'A+', 'B', 'C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+']
tmp = [k.lower() for k in key2]
key2 += tmp
for k, i in zip(key2, range(24)):
    if k not in inv_key_map:
        inv_key_map[k] = i
