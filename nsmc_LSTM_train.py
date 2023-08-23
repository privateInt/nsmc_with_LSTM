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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib


def main():       
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
        
    with open("y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
        
    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
            
    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    
    # 7.LSTM 모델링
    with tf.device('/device:GPU:3'): # GPU 설정
        embedding_dim = 100
        hidden_units = 128

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

if __name__ == '__main__':
    main()