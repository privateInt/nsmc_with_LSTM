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
    with open("X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
        
    with open("y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    max_len = 30

    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
    
if __name__ == "__main__":
    main()