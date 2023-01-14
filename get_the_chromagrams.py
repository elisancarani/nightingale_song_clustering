#!pip install librosa
from pathlib import Path
import os
import pandas as pd
import librosa
import librosa.display
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt



#constants
SR = 22050

DIR = './dataset/songs_extracted'

#function that loads the songs from the songs_extracted dataset
def load_songs_extracted(directory):
    data = []
    for bird_names in os.listdir(directory)[:20]:
        path_bird_names = os.path.join(directory, bird_names)
        if os.path.isdir(path_bird_names):
            for channel in os.listdir(path_bird_names):
                if channel == "CH2":
                    path_channel  = os.path.join(path_bird_names, channel)
                    if os.path.isdir(path_channel):
                        for song in  os.listdir(path_channel):
                            song_num = int(re.search('\d+', song).group())
                            path_song = os.path.join(path_channel, song)
                            signal, sr = librosa.load(path_song)
                            data.append([bird_names, song_num, sr, signal]) #channel is CH2 by default

    # return dataframe creation
    return pd.DataFrame(data, columns=['bird', 'song', 'sr', 'signal'])


df = load_songs_extracted(Path(DIR))
print("Done extracting songs")
print()

df['signal_length'] = df['signal'].apply(lambda x: len(x))
mean = np.mean(df['signal_length'])
mean = int(mean)
print('mean:', mean)
print('standard deviation:', np.std(df['signal_length']))
print(df[df['signal_length'] < 100000]['signal_length'].count(), 'signals are not truncated out of', df['signal_length'].count())
print()

policy = 'truncation' #'padding'

if policy == 'padding':
    maxim = np.max(df['signal_length'])
    print(maxim)
    df['padded_signal'] = df['signal'].apply(lambda x: np.pad(x, (0, maxim-len(x))) )
    
elif policy == 'truncation':
    #TODO truncation
    trunc_thresh = 11*22050  #maxim
    #padding
    df['padded_signal'] = df['signal'].apply(lambda x: np.pad(x, (0, trunc_thresh-len(x))) if len(x)<trunc_thresh else x) 
    #truncation
    df['padded_signal'] = df['padded_signal'].apply(lambda x: x[0:trunc_thresh] if len(x)>trunc_thresh else x) 
    
print(len(df['padded_signal'][1]))


sr = SR     
order = 5
lf = 1500
filtered = []


def high_pass(sig):
    nyq = sr/2
    sos = butter(order, [lf/nyq], analog=False, btype='highpass', output='sos')
    return sosfiltfilt(sos,sig)


df['filtered_signal'] = df['padded_signal'].apply(lambda x: high_pass(x))    


chromagrams = []
for signal in df['filtered_signal']:
    chrom = librosa.feature.chroma_stft(y=signal, sr=SR)
    chromagrams.append(chrom)


plt.figure(figsize=(30, 10))
for i in range(len(chromagrams)):
	librosa.display.specshow(chromagrams[i], x_axis='time', y_axis='chroma', hop_length=512, cmap='coolwarm')
	plt.savefig("chromagrams/" + str(i) + "out.png", bbox_inches='tight', facecolor='white')
	plt.clf()
	if i % 50 == 0:
		print("Made up to number" + str(i))