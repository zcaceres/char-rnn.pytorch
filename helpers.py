# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
import torchaudio
import glob
from pathlib import Path

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

MAX_AUDIO_FILES = 10

def read_audio_dir(dir_name):
  p = Path(dir_name)
  wavs = glob.glob(str(p) + '/**/*.wav', recursive=True)
  print('Found ' + str(len(wavs)) + ' files')
  final_audio = torch.tensor()
  # Concat all audio together
  for wav in wavs[:MAX_AUDIO_FILES]:
    sig, sr = torchaudio.load(wav)
    final_audio = torch.cat((final_audio, sig), 0)
  return final_audio, final_audio.shape[0]

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

