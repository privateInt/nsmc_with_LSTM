import streamlit as st
from PIL import Image
import requests
import json
from json import JSONEncoder
import torch
import numpy as np
import time
import os, cv2
from glob import glob
import pandas as pd
import numpy as np
import base64


def main():
    # device = 'cuda:0'
    # model = load_torch()
    
    # model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained('kakao_dictionary')
    st.title("광고 copy generation")

    # st.write("Enter a sentence to get similar sentences")
    
    with st.form("input_form"):
        boxholder = st.empty()
        submitplace = st.empty()
        

        c1, c2 = boxholder.columns([0.5, 0.5])
        
        d1, d2 = submitplace.columns([0.83, 0.18])

        # comp_message = c1.text_input("브랜드명을 넣어주세요")
        category_message = c1.text_input("카테고리 명을 넣어주세요")
        comp_message = c1.text_input("회사명을 넣어주세요")
        brand_message = c1.text_input("브랜드명을 넣어주세요")
        name_message = c1.text_input("제품명을 넣어주세요")
        key_message = c1.text_input("key를 넣어주세요")
        # model_num = c2.number_input("모델 번호를 넣어주세요", min_value=1, max_value=9, value=9)
        
        submitted = d2.form_submit_button("Submit")

#     print(seg_choice)
    
    # st.write(type(category_message), comp_message, brand_message, name_message,key_message)
    
#     if b2.button("show result", key='message'):
    if submitted:
        to_send = {'[CAT]': category_message, '[COMP]': comp_message, '[BRAND]': brand_message, '[NAME]': name_message, '[KEY]': key_message}
        response = requests.post('http://localhost:8000/copy', json.dumps(to_send))
        output = response.json()
        for i in output:
            st.markdown(i)
        # for j in gen_ids:
        #     generated = tokenizer.decode(j)
        #     st.markdown("**{}**".format(generated.split('[EOS]')[0].split('[COPY]')[1]))

            
            
# @st.cache(allow_output_mutation=True)
# def load_torch():
#     print('start loading')
#     model = AutoModelForCausalLM.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   pad_token_id='[EOS]',
#   torch_dtype='auto')
#     print('loading model dict')
#     model.load_state_dict(torch.load('pytorch_model.bin'))
#     model.eval()
#     return model

if __name__ == '__main__':

    main()