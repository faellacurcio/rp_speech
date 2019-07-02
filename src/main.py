from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from time import sleep
import random
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

AUDIO_PATH = "./src/audio_samples/**/"

def get_folders():
    folders = []
    for it_folder in glob.glob(AUDIO_PATH):
        # print(it_folder.split("/")[3])
        # print(it_folder)
        folders.append(it_folder)
    return folders

def get_subfolders(path):
    subfolders = []
    for it_subfolder in glob.glob(path+"*"):
        subfolders.append(it_subfolder)
    return subfolders

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

def get_training_from_folder(path):
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

def get_training_from_file(path):
    """
    input:
        path = string com o nome do arquivo de treinamento
    output: 
        vetor com os valores de MFCC para todos os audios dentro da pasta
    """
    # Variavel que armazena os dados da primeira pessoa
    person_data = []

    # Para cada arquivo na pasta de treino da pessoa 1 coleta os valores de MFCC
    print("Searching file: "+"./"+str(path))
    # Tamanho do bloco
    CHUNK = 1024

    # Abre o arquivo
    (rate,sig) = wav.read(path)

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


# =======================================================

# Train Person 0
print("Getting data from person 1")
Person0_data = get_training_from_folder("rp_speech/src/Train2")
#print(Person0_data)

# Train Person 1
print("Getting data from person 2")
Person1_data = get_training_from_folder("rp_speech/src/Train1")
#print(Person1_data)


X = np.array([])
Y = np.array([])
y_counter = 0;
for person in get_folders():
    subfolder = random.choice(get_subfolders(person))
    training_audio = random.choice(get_subfolders(subfolder))

    X = np.concatenate((X,get_training_from_file(training_audio)),axis=0)
    y = np.concatenate((y, counter*np.ones([1,len(training_audio)]))), axis=1)
    counter+=1

    

#for folder in glob.glob("./src/audio_samples/**/"):

quit()
print("")
# ---------------------------------------- MLP -----------------------------------------
start = time.time()
print("Fit MLP")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,3), random_state=1)
clf.fit(X, np.ravel(y))
print("classify MLP")
MLP_result = play_classify("rp_speech/src/Test/10SecsInterval2", clf)
end = time.time()
print([(end - start)/60, " minutes"])
print("")
# ---------------------------------------- SVC -----------------------------------------
start = time.time()
print("Fit SVC")
clf = svm.SVC(gamma='scale')
clf.fit(X, np.ravel(y))
print("classify SVC")
SVC_result = classify("rp_speech/src/Test/10SecsInterval2", clf)
end = time.time()
print([(end - start)/60, " minutes"])
print("")
# ---------------------------------------- QDA -----------------------------------------
start = time.time()
print("Fit QDA")
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, np.ravel(y))
print("classify QDA")
QDA_result = classify("rp_speech/src/Test/10SecsInterval2", clf)
end = time.time()
print([(end - start)/60, " minutes"])
print("")
# ---------------------------------------- Random Florest -----------------------------------------
start = time.time()
print("Fit Random Florest")
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X, np.ravel(y))
print("classify Random Florest")
RFC_result = classify("rp_speech/src/Test/10SecsInterval2", clf)
end = time.time()
print([(end - start)/60, " minutes"])
print("")
# ---------------------------------------- Expected Answer -----------------------------------------
aux = np.concatenate(( 10*[0], 10*[1]), axis=0)
aux2 = list(np.tile(aux, (1, 24)))
expected_output =  np.concatenate(( aux2[0],2*[1]), axis=0)

print("expected_output", expected_output)
print("MLP_result", MLP_result)
print("SVC_result", SVC_result)
print("QDA_result", QDA_result)
# print("SVM_result", SVM_result)
print("RFC_result", RFC_result)

# Saving the objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump([expected_output, MLP_result,SVC_result,QDA_result,RFC_result], f)
    #pickle.dump([expected_output, MLP_result], f)

