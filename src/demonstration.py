from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from record import RECORD
from time import sleep
import pickle
import glob
import os

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


import pyaudio  
import wave 
import time

# COnvert one file
# sox example.flac -r 44100 -c 1 example3.wav

# COnvert batch
# for old in *.flac; do sox $old -r 44100 -c 1 `basename $old .flac`.wav; done

# Concatenate Batch
# https://trac.ffmpeg.org/wiki/Concatenate
# ffmpeg -f concat -safe 0 -i <(find . -name '*.wav' -printf "file '$PWD/%p'\n") -c copy output.wav

# Split audio
# ffmpeg -i Person1_COMPLETE.wav -f segment -segment_time 10 -c copy out%03d.wav



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
        # try:
        #     mfcc_values = mfcc_values[] 
        # except:
        #     mfcc_values = mfcc_values[]
            
        print(mfcc_values)
        stream.write(data)
        data = f.readframes(CHUNK)  
    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate() 

def play_classify(path, clf):
    #define stream chunk   
    CHUNK = 1024  
    time_counter = 1
    person_guess = []
    result = []

    #open a wav format music  
    print("Opening file:")
    print("./"+path+".wav")
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
        aux = next(mfcc_sig_splits)
        aux2 = mfcc(aux,samplerate = rate, nfft=1104)
        person_guess.append(clf.predict(aux2))
        stream.write(data)
        data = f.readframes(CHUNK)  
        if(len(person_guess) == 43):# every 43 blocks is 1 second for 44100Hz rate
            boolean_check = np.argmax([person_guess.count(0), person_guess.count(1)])
            if(boolean_check):
                print("person0")
            else:
                print("person1")
            print("---------"+str(time_counter)+" s")
            time_counter+=1
            person_guess = []
            result.append(0 if boolean_check else 1)
    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate() 
    return result

def classify(path, clf):
    #define stream chunk   
    CHUNK = 1024
    time_counter = 1
    person_guess = []
    result = []

    #open a wav format music  
    print("Opening file:")
    print("./"+path+".wav")
    f = wave.open("./"+path+".wav","rb")
    #open a wav format mfcc
    (rate,sig) = wav.read("./"+path+".wav")

    mfcc_sig_splits = chunks(sig,CHUNK)
 
    try:
        while True:
            aux = next(mfcc_sig_splits)
            aux2 = mfcc(aux,samplerate = rate, nfft=1104)
            person_guess.append(clf.predict(aux2))
            if(len(person_guess) == 43):# every 43 blocks is 1 second for 44100Hz rate
                boolean_check = np.argmax([person_guess.count(0), person_guess.count(1)])
                # if(boolean_check):
                #     print("person0")
                # else:
                #     print("person1")
                # print("---------"+str(time_counter)+" s")
                time_counter+=1
                person_guess = []
                result.append(0 if boolean_check else 1)
    except:
        pass

    return result


def get_training_data(path):
    """
    input:
        path = string com o nome da pasta de treinamento
    output: 
        vetor com os valores de MFCC para todos os audios dentro da pasta
    """
    # Variavel que armazena os dados da primeira pessoa
    person_data = []

    # Para cada arquivo na pasta de treino da pessoa 1 coleta os valores de MFCC
    print("Searching files: "+"./"+str(path)+"/*.wav")
    for file in glob.glob("./"+str(path)+"/*.wav"):
        # Tamanho do bloco
        CHUNK = 1024

        # Abre o arquivo
        f = wave.open(str(file),"rb")
        (rate,sig) = wav.read(file)

        #Separa os dados em blocos de tamanho CHUNK
        mfcc_sig_splits = chunks(sig,CHUNK)
        # d_mfcc_feat = delta(mfcc_feat, 2)

        # Enquanto houver dados, append na variavel de dados
        while True:
            try:
                aux = next(mfcc_sig_splits)
                mfcc_values = mfcc(aux,samplerate = rate, nfft=1104)
                person_data.append(list(mfcc_values[0]))
            except:
                break
    
    return person_data

obj_record = RECORD(THRESHOLD = 10000)

# =======================================================

# Train Person 1
print("Getting data from person 1")
Person0_data = get_training_data("rp_speech/src/Train0d")
#print(Person0_data)

# Train Person 2
print("Getting data from person 2")
Person1_data = get_training_data("rp_speech/src/Train1")
#print(Person1_data)



X = np.concatenate((Person0_data,Person1_data),axis=0)
y = np.concatenate((np.ones([1,len(Person0_data)]),np.zeros([1,len(Person1_data)])), axis=1)
print("")
# ---------------------------------------- MLP -----------------------------------------
start = time.time()
print("Fit MLP")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,3), random_state=1)
clf.fit(X, np.ravel(y))
print("classify MLP")
MLP_result = play_classify("rp_speech/src/Test/10SecsInterval", clf)
end = time.time()
print([(end - start)/60, " minutes"])
print("")