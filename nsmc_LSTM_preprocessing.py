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
    # 1.데이터 로드
    root_path = '/home/ubuntu/workspace/seunghoon/z_dataset/nsmc_dataset/nsmc'
    
    train_data = pd.read_table(os.path.join(root_path, 'ratings_train.txt'))
    test_data = pd.read_table(os.path.join(root_path, 'ratings_test.txt'))
    
    # 2.데이터 정제
    train_data['document'].nunique(), train_data['label'].nunique() # 중복제거
    train_data.drop_duplicates(subset=['document'], inplace=True) # document 열의 중복 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) # 한글과 공백을 제외하고 모두 제거
    train_data['document'] = train_data['document'].str.replace('^ +', "", regex=True) # white space 데이터를 empty value로 변경
    train_data['document'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how = 'any')
    
    test_data['document'].nunique(), test_data['label'].nunique() # 중복 제거
    test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) # 정규 표현식 수행
    test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True) # 공백은 empty 값으로 변경
    test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거
    
        
    # 3.토큰화
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    okt = Okt()
    
    X_train = []
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_train.append(stopwords_removed_sentence)
        
    X_test = []
    for sentence in tqdm(test_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_test.append(stopwords_removed_sentence)
        
    # 4.정수 인코딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
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
    
    tokenizer = Tokenizer(vocab_size) 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])
    
    # 5.빈 샘플 제거
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    
    # 6.패딩
    max_len = 30
    
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    # 7. save pickle    
    object_lst = [train_data, test_data, tokenizer, X_train, X_test, y_train, y_test]
    name_lst = ["train_data.pkl", "test_data.pkl", "tokenizer.pkl", "X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
    
    for name, object_name in zip(name_lst, object_lst):
        with open(name, "wb")as f:
            pickle.dump(object_name, f)

    
if __name__ == "__main__":
    main()
    