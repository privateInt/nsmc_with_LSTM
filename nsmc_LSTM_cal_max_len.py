import re
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib

def main():
    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
    print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
    # plt.hist([len(review) for review in X_train], bins=50)
    # plt.xlabel('length of samples')
    # plt.ylabel('number of samples')
    # plt.show()
    
    def below_threshold_len(max_len, nested_list): # 검증
        count = 0
        for sentence in nested_list:
            if(len(sentence) <= max_len):
                count = count + 1
        print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))
        
    max_len = 30
    below_threshold_len(max_len, X_train)
    
if __name__ == "__main__":
    main()