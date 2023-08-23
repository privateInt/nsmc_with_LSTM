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
#     # 1.데이터 로드
#     root_path = '/home/ubuntu/workspace/seunghoon/z_dataset/nsmc_dataset/nsmc'
    
#     train_data = pd.read_table(os.path.join(root_path, 'ratings_train.txt'))
#     test_data = pd.read_table(os.path.join(root_path, 'ratings_test.txt'))
    
#     # 2.데이터 정제
#     train_data['document'].nunique(), train_data['label'].nunique() # 중복제거
#     train_data.drop_duplicates(subset=['document'], inplace=True) # document 열의 중복 제거
#     train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) # 한글과 공백을 제외하고 모두 제거
#     train_data['document'] = train_data['document'].str.replace('^ +', "", regex=True) # white space 데이터를 empty value로 변경
#     train_data['document'].replace('', np.nan, inplace=True)
#     train_data = train_data.dropna(how = 'any')
    
#     with open("train_data.pkl", "wb") as f:
#         pickle.dump(train_data, f)
    
#     test_data['document'].nunique(), test_data['label'].nunique() # 중복 제거
#     test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
#     test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) # 정규 표현식 수행
#     test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True) # 공백은 empty 값으로 변경
#     test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
#     test_data = test_data.dropna(how='any') # Null 값 제거
    
#     with open("test_data.pkl", "wb") as f:
#         pickle.dump(test_data, f)


    with open("train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
        
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    
    # 3.토큰화
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    okt = Okt()
    
    X_train = []
#     for sentence in tqdm(train_data['document']):
#         tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
#         stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
#         X_train.append(stopwords_removed_sentence)
        
#     with open("X_train.pkl", "wb")as f:
#         pickle.dump(X_train, f)
        
    with open("X_train.pkl", "rb")as f:
        X_train = pickle.load(f)

    X_test = []
#     for sentence in tqdm(test_data['document']):
#         tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
#         stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
#         X_test.append(stopwords_removed_sentence)
        
#     with open("X_test.pkl", "wb")as f:
#         pickle.dump(X_test, f)
        
    with open("X_test.pkl", "rb")as f:
        X_test = pickle.load(f)
        
    # 4.정수 인코딩
#     tokenizer = Tokenizer()
    
    # with open('tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle)
        
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    tokenizer.fit_on_texts(X_train)
    
    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq += value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt += 1
            rare_freq += value
            
    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    
    tokenizer = Tokenizer(vocab_size) 
        
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])
    
    # 5.빈 샘플(empty samples) 제거
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    
    # 6.패딩
    # print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
    # print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
    # plt.hist([len(review) for review in X_train], bins=50)
    # plt.xlabel('length of samples')
    # plt.ylabel('number of samples')
    # plt.show()
    
    # def below_threshold_len(max_len, nested_list):
    #     count = 0
    #     for sentence in nested_list:
    #         if(len(sentence) <= max_len):
    #             count = count + 1
    #     print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))
    
    max_len = 30 # 길이 분포를 통해 max_len을 30으로 설정
    # below_threshold_len(max_len, X_train) # 샘플 길이가 30이하인 샘플의 비율 구하기
    
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
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