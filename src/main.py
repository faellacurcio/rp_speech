# "Trains a different classificator and compare result"

import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from record import RECORD
from time import sleep

import pyaudio  
import wave 

# database https://github.com/Jakobovski/free-spoken-digit-dataset/tree/master/recordings


def bip():
    "Beeps for indicate code progress"
    duration = 0.1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def chunks(data, size):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def play_wav(path):
    #define stream chunk   
    CHUNK = 1024  

    #open a wav format music  
    f = wave.open("./"+path+".wav","rb")
    #open a wav format mfcc
    (rate,sig) = wav.read("./"+path+".wav")

    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(CHUNK)  

    mfcc_sig_splits = chunks(sig,CHUNK)

    #play stream  
    while data:
        # sox example.flac -r 44100 -c 1 example3.wav
        # mfcc_feat = mfcc(next(mfcc_sig_splits),f.getframerate())
        # next(mfcc_sig_splits)
        mfcc_values = next(mfcc_sig_splits)
        mfcc_values = mfcc_values[:,1]
        print(mfcc_values)
        stream.write(data)
        data = f.readframes(CHUNK)  
    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate() 

obj_record = RECORD(THRESHOLD = 10000)

print("Pessoa 1 fala")
sleep(1)
# bip()
# /home/rafael/Desktop/RP_speech/audio_samples/174/84280/174-84280-0001.flac
personfile1 = "person1"

# dr1, data1 = obj_record.record_to_raw_audio(personfile1)
# obj_record.saveToFile(personfile1)
play_wav(personfile1)
# bip()
# bip()

# print("Pessoa 2 fala")
# sleep(1)
# bip()
# # /home/rafael/Desktop/RP_speech/audio_samples/84/121123/84-121123-0005.flac
# personfile2 = "person2"
# # dr2, data2 = obj_record.record_to_raw_audio(personfile2)
# obj_record.saveToFile(personfile2)
# bip()

(rate,sig) = wav.read(personfile1+".wav")
mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)

print(mfcc_feat.shape)