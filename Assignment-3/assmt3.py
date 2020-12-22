import librosa
import numpy as np
import os
import pickle
import hmmlearn
import itertools

from hmmlearn import hmm
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math
import scipy.misc
from scipy.io.wavfile import read
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from scipy.io import wavfile


def end_pointing(filepath):
    a,sr= librosa.load(filepath)

    #### Slicing / extracting frames
    ### 20ms window frame

    i = np.int(len(a) )
    sliced = []
    k = 0
    while k<= (i/160)-2:
        m = a[k*160 : (k+2)*160]
        sliced.append(m)
        k = k+1
    # print(len(sliced))

    ### finding the start and using the energy
    s =0 
    l = 0
    noise = 1 
    updated = 0
    e = np.int(len(sliced))-1

    while l<np.int(len(sliced)):
        #### hamming window
        h = np.hamming(320)
        en = np.matmul(sliced[l]*h,sliced[l]*h)
        #### start point selection
        if(en > 0.004):
            if(noise == 1):
                noise = 0
                if updated == 0:
                    updated = 1
                    s = l
        ##end point selection 
        if(en < 0.00001 and noise == 0):
            noise = 1
            e = l
            ## following can be used for training cases with large silence after utterance
            # if e-s > 50 :
            #     noise = 1

        l = l+1

    # print("start" ,s , "end " ,e)    

    d = e-s
    if d < 50:
        e = np.int(len(sliced))-1

    end_pointed = np.array(a[s*160 : (e+2)*160])
    return end_pointed,sr


###################################################################
###### ------>  end pointing all files

os.listdir('Commands_Dataset/train')

for word in os.listdir('Commands_Dataset/train'): 
    print(word)
    for sample in os.listdir('Commands_Dataset/train/'+ word): 
        # print(word, sample)
        endp , samp = end_pointing('Commands_Dataset/train/'+ word + '/' + sample)
        ### to ignoise only empyty / noise files
        if np.int(len(endp)) >10000 :
            sf.write('Commands_Dataset/train_end_pointed/'+ word + '/' + sample, endp, samp)



#####################---> pre emphasis

# pre emphasis
high_pass_output = []
i = 1

def pre_emp(filepath):
    high_pass_output = []
    i = 1
    input1,sr= librosa.load(filepath)
    high_pass_output.append(np.float(input1[0]))
    while i< np.int(len(input1)):
        a = np.float(input1[i]-0.95*input1[i-1])
        high_pass_output.append(a)
        i = i+1
    return high_pass_output,sr



os.listdir('Commands_Dataset/train_end_pointed')

for word in os.listdir('Commands_Dataset/train_end_pointed'): 
    print(word)
    for sample in os.listdir('Commands_Dataset/train_end_pointed/'+ word): 
        print(word, sample)
        pre_emph, samp = pre_emp('Commands_Dataset/train_end_pointed/'+ word + '/' + sample)
        sf.write('Commands_Dataset/train_pre_emp/'+ word + '/' + sample, pre_emph, samp)
    

with open("training_pre_emp.pkl", "wb") as file:
    pickle.dump(training , file)

###############################################################################################################################

## ---> MFCC feature extraction

def mfcc(filepath):
    a ,samp = librosa.load(filepath)
    #mfcc
    mfcc = librosa.feature.mfcc(y=a, sr=samp, n_mfcc=13 ,S=None, dct_type=2, norm='ortho')
    ## difference of mfcc
    mfcc_delta = librosa.feature.delta(mfcc,order=1, mode = 'nearest')
    ## difference of difference
    mfcc_delta2 = librosa.feature.delta(mfcc,order=2, mode = 'nearest')

    mfcc_features = np.concatenate((mfcc, mfcc_delta , mfcc_delta2 ), axis=0)
    
    return mfcc_features 


################################################################################################################
# # --> loading training and test data .

##### -->  training data .

training = {}
for word in os.listdir('Commands_Dataset/train_pre_emp'):
    for sample in os.listdir('Commands_Dataset/train_pre_emp/'+word):
        true_word = word
        p = 'Commands_Dataset/train_pre_emp/'+word + '/' + sample
        print(p)
        sample_mfcc = mfcc(p).T
        
        #### adding to training data dict
        if true_word not in training.keys():
            training[true_word] = []
            training[true_word].append(sample_mfcc)
        else:
            y = training[true_word]
            y.append(sample_mfcc)
            training[true_word] = y


# ####################---> saving training data

with open("training_pre_emp_noisy.pkl", "wb") as file:
    pickle.dump(training , file)


def pre_emp_f(filesignal):
    high_pass_output = []
    i = 1
    input1 = filesignal
    high_pass_output.append(np.float(input1[0]))
    while i< np.int(len(input1)):
        a = np.float(input1[i]-0.95*input1[i-1])
        high_pass_output.append(a)
        i = i+1
    return high_pass_output

def mfccf(filesignal,sam):
    # print(type(filesignal))
    a = np.array(filesignal)
    #mfcc
    samp = sam
    mfcc = librosa.feature.mfcc(y=a, sr=samp, n_mfcc=13 ,S=None, dct_type=2, norm='ortho')
    ## difference of mfcc
    mfcc_delta = librosa.feature.delta(mfcc,order=1, mode = 'nearest')
    ## difference of difference
    mfcc_delta2 = librosa.feature.delta(mfcc,order=2, mode = 'nearest')

    mfcc_features = np.concatenate((mfcc, mfcc_delta , mfcc_delta2 ), axis=0)
    
    return mfcc_features 

############ --> task A testing data 

testing_clean = {}
for word in os.listdir('Commands_Dataset/test_clean'):
    for sample in os.listdir('Commands_Dataset/test_clean/'+word):
        true_word = word
        p = 'Commands_Dataset/test_clean/'+word + '/' + sample
        enp, sa  = end_pointing(p)
        pre_empha = pre_emp_f(enp)  
        print(p)
        sample_mfcc = mfccf(pre_empha,sa).T
        
        #### adding to testing data dict
        if true_word not in testing_clean.keys():
            testing_clean[true_word] = []
            testing_clean[true_word].append(sample_mfcc)
        else:
            y = testing_clean[true_word]
            y.append(sample_mfcc)
            testing_clean[true_word] = y

print('testingA data created')
with open("testA.pkl", "wb") as file:
    pickle.dump(testing_clean , file)



############ --> task B testing data
testing_noisy = {}
for word in os.listdir('Commands_Dataset/test_noisy'):
    for sample in os.listdir('Commands_Dataset/test_noisy/'+word):
        true_word = word
        p = 'Commands_Dataset/test_noisy/'+word + '/' + sample
        enp, sa  = end_pointing(p)
        pre_empha = pre_emp_f(enp)  
        print(p)
        sample_mfcc = mfccf(pre_empha,sa).T

        #### adding to testing data dict
        if true_word not in testing_clean.keys():
            testing_clean[true_word] = []
            testing_clean[true_word].append(sample_mfcc)
        else:
            y = testing_clean[true_word]
            y.append(sample_mfcc)
            testing_clean[true_word] = y

print('testingB data created')
with open("testB.pkl", "wb") as file:
    pickle.dump(testing_noisy , file)


training_data_file  =  open('training_pre_emp_noisy.pkl', 'rb')
training = pickle.load(training_data_file)
training_data_file.close()

# # ###################### training using hmmlearn
trained_model = {}
hmm_states = 6

for true_word in training.keys():
    # model = hmm.GMMHMM(n_components=hmm_states, covariance_type='diag', n_iter=10 , verbose=False )
    model = hmm.GaussianHMM(n_components = hmm_states, covariance_type='full', n_iter=10)
    print(true_word)
    ## collecting the features of each word to train 
    current_training_data = training[true_word]
    # print(current_training_data)

    u = np.zeros([len(current_training_data), ], dtype=np.int)

    q = 0
    while q < np.int(len(current_training_data)):
        u[q] = current_training_data[q].shape[0]
        q =q+1
    
    current_training_data = np.vstack(current_training_data)
    #### this fit() trains the model
    model.fit(current_training_data, lengths=u)
    
    trained_model[true_word] = model


# ################################ ---> saving model

with open("trained.pkl", "wb") as file:
    pickle.dump(trained_model, file)

# # ########################### --->  confusion matrix creation

testA_model_file  =  open('testA.pkl', 'rb')
testingA = pickle.load(testA_model_file)
testA_model_file.close()

testB_model_file  =  open('testB.pkl', 'rb')
testingB = pickle.load(testB_model_file)
testB_model_file.close()

hmm_model_file  =  open('trained.pkl', 'rb')
trained_model_hmm = pickle.load(hmm_model_file)
hmm_model_file.close()


# ##################### making the input featurs and output labels of test data  

#### --- > Task A
x_testA = []
y_testA = []
for a in testingA.keys():
    featu = testingA[a]
    l = np.int(len(featu))
    for t  in range(l):
        y_testA.append(a)
        x_testA.append(featu[t])



#### --- > Task B
x_testB = []
y_testB = []
for a in testingB.keys():
    featu = testingB[a]
    l = np.int(len(featu))
    for t  in range(l):
        y_testB.append(a)
        x_testB.append(featu[t])

print("done test data creation")


# ############ prediction function for given input and trained model

def prediction(test_data, trained):
    predicted_label = []
    words = []
    score_values = []

    for k in trained.keys():
        score_values.append(trained[k].score(test_data))
        words.append(k)
    predicted_label.append(score_values.index(max(score_values)))
    return words[predicted_label[0]]



n = len(x_testA)
pred_test = []
pred_train = []
tot_test = 0

for i in range(n):
  y_pred = prediction(x_testA[i],trained_model_hmm )
  if y_pred == y_testA[i]:
      tot_test += 1
  pred_test.append(y_pred)


n = len(x_testB)
pred_testB = []
pred_train = []
tot_test = 0

for i in range(n):
  y_pred = prediction(x_testB[i],trained_model_hmm )
  if y_pred == y_testB[i]:
      tot_test += 1
  pred_testB.append(y_pred)



with open("predtest.pkl", "wb") as file:
    pickle.dump(pred_test, file)

with open("ytestA.pkl", "wb") as file:
    pickle.dump(y_testA, file)
  
with open("ytestB.pkl", "wb") as file:
    pickle.dump(y_testB, file)

with open("predtestB.pkl", "wb") as file:
    pickle.dump(pred_testB, file)




trained_model_file  =  open('predtest.pkl', 'rb')
pred_test = pickle.load(trained_model_file)
trained_model_file.close()


trained_model_file  =  open('predtestB.pkl', 'rb')
pred_testB = pickle.load(trained_model_file)
trained_model_file.close()


trained_model_file  =  open('ytestA.pkl', 'rb')
y_testA = pickle.load(trained_model_file)
trained_model_file.close()

trained_model_file  =  open('ytestB.pkl', 'rb')
y_testB = pickle.load(trained_model_file)
trained_model_file.close()


cm = confusion_matrix(y_testA, pred_test)


print(type(cm))
print(np.diag(cm))
accuray = 0.0
accuray = sum(np.diag(cm)) / (sum(sum(cm)))
print('Accuracy = ',accuray*100,'%')

classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
normalize=False
title='Confusion matrix -- Clean test Data --Noisy Train data '
cmap=plt.cm.Blues

plt.subplot(1,2,1)
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
# plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('TaskA-noisy-train-gauss.png', dpi = 200)
# # plt.show()


cm = confusion_matrix(y_testB, pred_testB)
print(type(cm))
print(np.diag(cm))
accuray = 0.0
accuray = sum(np.diag(cm)) / (sum(sum(cm)))
print('Accuracy = ',accuray*100,'%')

classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
normalize=False
title='Confusion matrix -- Noisy test Data --Noisy train data'
cmap=plt.cm.Blues

plt.subplot(1,2,2)
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
# plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('TaskB-noisy-train-gauss.png', dpi = 200)
plt.show()