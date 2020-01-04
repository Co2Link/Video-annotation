import glob
import os
import csv
import pandas as pd
from utils import timethis

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score

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

def train_multi():
    pass

def data_preprocess(inputs,labels):
    """ format data from dict"""
    columns = ['confidence','success','AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    action_name = 'smile'
    X = pd.DataFrame()
    Y = pd.Series()
    for video in ['1','2']:
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

        print(XY.label.value_counts())
        print('*** after upsampling. num of train({}) ,num of test({})***'.format(len(X_train),len(X_test)))

    

    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    loss = model.loss_curve_
    accuracy = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    recall = recall_score(y_test,pred)

    return loss,accuracy,f1,recall

@timethis
def main():
    X,Y = data_preprocess(read_input(),read_label())
    loss_list,accuracy_list,f1_list,recall_list=[],[],[],[]
    for i in tqdm(range(ITER)):
        loss,accuracy,f1,recall = train(X,Y,scale=SCALE,upsample=UPSAMPLE,test_ratio=TEST_RATIO,hidden_layer_sizes=NET_SIZE)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        recall_list.append(recall)
    
    
    for idx in range(len(loss_list)):
        print('accuracy: {}, f1: {}, recall: {}'.format(round(accuracy_list[idx],3),round(f1_list[idx],3),round(recall_list[idx],3)))
    print('avg: ',round(sum(accuracy_list)/len(accuracy_list),3),round(sum(f1_list)/len(f1_list),3),round(sum(recall_list)/len(recall_list),3))
        
    for loss in loss_list:
        plt.plot(loss)
        plt.show()
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter','--iteration',type=int,default=10)
    parser.add_argument('--scale',action='store_true')
    parser.add_argument('--upsample',action='store_true')
    parser.add_argument('--test_ratio',type=float,default=0.2)
    parser.add_argument('--net_size',type=str,default='10-10')
    args = parser.parse_args()

    ITER = args.iteration
    SCALE = args.scale
    UPSAMPLE = args.upsample
    TEST_RATIO = args.test_ratio
    NET_SIZE = tuple([int(num) for num in args.net_size.split('-')])
    
    main()

    print(""" argument """)
    print(args)