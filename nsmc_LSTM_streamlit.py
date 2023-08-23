import re
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from konlpy.tag import Okt
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib


st.title("영화 리뷰 감성 분석 테스트")

@st.cache(allow_output_mutation=True)
def call_model(model_name):
    loaded_model = load_model(model_name)
    return loaded_model

@st.cache(allow_output_mutation=True)
def call_okt():
    okt = Okt()
    return okt

@st.cache(allow_output_mutation=True)
def call_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

loaded_model = call_model('best_model.h5')
okt = call_okt()
tokenizer = call_tokenizer()

input_text = st.text_input("감성 분석할 문장을 입력 후 엔터를 눌러주세요.")

if input_text:
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    new_sentence = input_text
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거

    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 30) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    
    if(score > 0.5):
        st.write("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        st.write("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
