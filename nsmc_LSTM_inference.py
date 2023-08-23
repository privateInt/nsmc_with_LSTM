import re
import pickle
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

def main(args):
    loaded_model = load_model(args.model_path)
    okt = Okt()
    max_len = 30     
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    
    new_sentence = args.input_sentence
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(loaded_model.predict(pad_new)) # 예측
    print(args.input_sentence)
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Arguments')
    
    parser.add_argument('-is', '--input-sentence', type = str, default = "이 영화 개꿀잼 ㅋㅋㅋ", help='input sentence for sentiment analysis.')
    parser.add_argument('-mp', '--model-path', type = str, default = 'best_model.h5', help='choose best model.')
    
    args = parser.parse_args() 
    
    main(args)
    
# python nsmc_LSTM_inference.py -is "너무 재미없는 영화입니다"