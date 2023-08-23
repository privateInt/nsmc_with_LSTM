from flask import Flask, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
import json
from datasets import Dataset
from transformers import AdamW, get_scheduler
import os
block_size = 64

app = Flask(__name__)

    
    
    
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained('kakao_dictionary')
print('model loading...')

model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto'
)
print('model state loading...')
model.load_state_dict(torch.load('/home/ubuntu/workspace/howon/copy_generation/final_epoch6.bin'))
model.half()
print('model loading end!')
model.eval()
model.to(device)




@app.route('/copy', methods=['POST'])
def home():
    req = json.loads(request.data)
    lst = ['[CAT]', '[COMP]', '[BRAND]', '[NAME]', '[KEY]']
    prompt = '[BOS]'
    for i in lst:
        if i not in req:
            req[i] = ''
        prompt+=i+req[i]
    prompt+='[COPY]'
    print(prompt)
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device, non_blocking=True)
    return_tokens = []
    while True:
        with torch.no_grad():
            gen_tokens = model.generate(tokens, do_sample=True, temperature=0.95, max_length=64, num_return_sequences=25)
            generated = tokenizer.batch_decode(gen_tokens)
            for item in set([i.split('[COPY]')[1].split('[EOS]')[0] for i in generated]):
                return_tokens.append(item)

        if len(return_tokens)>10:
            break
    

    print(return_tokens[:10])
    return json.dumps(return_tokens[:10])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')