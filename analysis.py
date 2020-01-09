import glob
import os
import csv
import pandas as pd
import pickle
from utils import timethis
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.externals import joblib

import argparse

@timethis
def read_label():
    labels = dict()
    paths = glob.glob('data/label/*.csv')
    for path in paths:
        file_name = os.path.basename(path)
        video_name,person_name,action_name = file_name.split('_')
        action_name = action_name.split('.')[0]
        if video_name not in labels:
            labels[video_name] = dict()
        if person_name not in labels[video_name]:
            labels[video_name][person_name] = dict()
        with open(path,newline='') as f:
            reader = csv.reader(f)
            labels[video_name][person_name][action_name] = pd.Series([ int(row[0]) for row in reader])
    return labels

@timethis
def read_input():
    inputs = dict()
    paths = glob.glob('data/input/*_*.csv')
    for path in paths:
        file_name = os.path.basename(path)
        video_name,person_name = file_name.split('_')
        person_name = person_name.split('.')[0]
        if video_name not in inputs:
            inputs[video_name] = dict()
        inputs[video_name][person_name] = pd.read_csv(path)
    return inputs

def data_preprocess(inputs,labels,videos = ['1','2','3','4']):
    """ format data from dict"""
    columns = ['confidence','success','AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    action_name = 'smile'
    X = pd.DataFrame()
    Y = pd.Series()

    for video in videos:
        for person in ['a','b','c']:
            X = pd.concat([X,inputs[video][person][columns]],ignore_index=True)
            Y = pd.concat([Y,labels[video][person][action_name]],ignore_index=True)

    # only keep sample that success == 1
    XY = pd.concat([X,Y.rename('label')],axis=1)
    XY = XY[XY.success == 1]

    X = XY.drop(['confidence','success','label'],axis=1)
    Y = XY.label

    return X,Y


@timethis
def train(X,Y,scale = True,upsample = True,test_ratio = 0.2,hidden_layer_sizes=[10,10]):
    if scale:
        X[['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']] /= 5
        print('*** scaling ***')
        print(X.describe())
    # split training set and testing set
    X_train,X_test,y_train,y_test = train_test_split(X,Y)

    print('*** num of train({}) ,num of test({})***'.format(len(X_train),len(X_test)))

    if upsample:
        XY = pd.concat([X_train,y_train],axis=1)
        true = XY[XY.label==1]
        false = XY[XY.label==0]
        true_upsampled = resample(true,replace=True,n_samples=len(false))
        XY = pd.concat([true_upsampled,false])
        X_train,y_train = XY.drop(['label'],axis=1),XY.label

        print('*** after upsampling. num of train({}) ,num of test({})***'.format(len(X_train),len(X_test)))

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,max_iter=MAX_ITER)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    loss = model.loss_curve_
    accuracy = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    recall = recall_score(y_test,pred)
    precision = precision_score(y_test,pred)

    pred_train = model.predict(X_train)

    accuracy_train = accuracy_score(y_train,pred_train)
    f1_train = f1_score(y_train,pred_train)
    recall_train = recall_score(y_train,pred_train)
    precision_train = precision_score(y_train,pred_train)

    return loss,accuracy,f1,recall,precision,accuracy_train,f1_train,recall_train,precision_train,model

@timethis
def train_multi():
    X,Y = data_preprocess(read_input(),read_label(),videos=VIDEOS)
    loss_list,accuracy_list,f1_list,recall_list,precision_list,accuracy_train_list,f1_train_list,recall_train_list,precision_train_list=[],[],[],[],[],[],[],[],[]
    for i in tqdm(range(ITER)):
        loss,accuracy,f1,recall,precision,accuracy_train,f1_train,recall_train,precision_train,model = train(X,Y,scale=SCALE,upsample=UPSAMPLE,test_ratio=TEST_RATIO,hidden_layer_sizes=NET_SIZE)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy_train_list.append(accuracy_train)
        f1_train_list.append(f1_train)
        recall_train_list.append(recall_train)
        precision_train_list.append(precision_train)
    
    
    
    for idx in range(len(loss_list)):
        print('accuracy: {}, f1: {}, recall: {},precision: {}'.format(round(accuracy_list[idx],3),round(f1_list[idx],3),round(recall_list[idx],3),round(precision_list[idx],3)))
        print('train accuracy: {}, f1: {}, recall: {}'.format(round(accuracy_train_list[idx],3),round(f1_train_list[idx],3),round(recall_train_list[idx],3),round(precision_train_list[idx],3)))
    print('avg: ',round(sum(accuracy_list)/len(accuracy_list),3),round(sum(f1_list)/len(f1_list),3),round(sum(recall_list)/len(recall_list),3),round(sum(precision_list)/len(precision_list),3))
    print('train avg: ',round(sum(accuracy_train_list)/len(accuracy_train_list),3),round(sum(f1_train_list)/len(f1_train_list),3),round(sum(recall_train_list)/len(recall_train_list),3),round(sum(precision_train_list)/len(precision_train_list),3))
        
    for loss in loss_list:
        plt.plot(loss)
        plt.show()

def train_and_save_model():
    inputs = read_input()
    labels = read_label()
    X,Y = data_preprocess(inputs,labels,videos=Tr)
    loss,accuracy,f1,recall,precision,accuracy_train,f1_train,recall_train,precision_train,model = train(X,Y,scale=SCALE,upsample=UPSAMPLE,test_ratio=TEST_RATIO,hidden_layer_sizes=NET_SIZE)
    print('accuracy: {}, f1: {}, recall: {},precision: {}'.format(round(accuracy,3),round(f1,3),round(recall,3),round(precision,3)))
    print('train: accuracy: {}, f1: {}, recall: {}, precision: {}'.format(round(accuracy_train,3),round(f1_train,3),round(recall_train,3),round(precision_train,3)))


    X,Y = data_preprocess(input,labels,videos=TRAIN_VIDEOS)
    pred = model.predict(X)
    accuracy = accuracy_score(Y,pred)
    f1 = f1_score(Y,pred)
    recall = recall_score(Y,pred)
    precision = precision_score(Y,pred)
    print('*** test result ***')
    print('accuracy: {}, f1: {}, recall: {}, precision: {}'.format(round(accuracy,3),round(f1,3),round(recall,3),round(precision,3)))

    if not os.path.exists('models'):
        os.mkdir('models')
        print('*** create directory /models ***')

    file_name = 'model' + str(int(time.time()))

    joblib.dump(model,'models/{}'.format(file_name))    
    print('*** saved model at /models/{}'.format(file_name))

    plt.plot(loss)
    plt.show()

def get_model():
    model = joblib.load('models/model')
    return model
def test_model():
    X,Y = data_preprocess(read_input(),read_label(),videos=VIDEOS)
    model = joblib.load('models/{}'.format(MODEL_NAME))
    pred = model.predict(X)
    accuracy = accuracy_score(Y,pred)
    f1 = f1_score(Y,pred)
    recall = recall_score(Y,pred)
    precision = precision_score(Y,pred)

    print('*** test result ***')
    print('accuracy: {}, f1: {}, recall: {}, precision: {}'.format(round(accuracy,3),round(f1,3),round(recall,3),round(precision,3)))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter','--iteration',type=int,default=5)
    parser.add_argument('--scale',action='store_true')
    parser.add_argument('--upsample',action='store_true')
    parser.add_argument('--test_ratio',type=float,default=0.2)
    parser.add_argument('--net_size',type=str,default='10-10')
    parser.add_argument('--max_iter',type=int,default=200)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--train_videos',type=str,default='123')
    parser.add_argument('--test_videos',type=str,default='4')
    parser.add_argument('--model_name',type=str,default='')
    args = parser.parse_args()

    ITER = args.iteration
    SCALE = args.scale
    UPSAMPLE = args.upsample
    TEST_RATIO = args.test_ratio
    NET_SIZE = tuple([int(num) for num in args.net_size.split('-')])
    MAX_ITER = args.max_iter
    TRAIN_VIDEOS = [str(i) for i in args.train_videos]
    TEST_VIDEOS = [str(i) for i in args.test_videos]
    MODEL_NAME = args.model_name

    if args.train:
        train_and_save_model()
    elif args.test:
        test_model()
    else:
        train_multi()

    print(""" argument """)
    print(args)