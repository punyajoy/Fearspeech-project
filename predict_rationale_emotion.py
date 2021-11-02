import pandas as pd
from glob import glob
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from model_code.rationale_model import *
from data_handler.tf_processor import  *
from model_code.model import *
from tqdm import tqdm
import time
import torch
import numpy as np
import pandas as pd
import os 
import random
import datetime
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,hamming_loss
from apiconfig import project_name,api_token
import neptune.new as neptune
import GPUtil
import pickle
    
model_memory=9
total_memory=16

def get_gpu(gpu_id):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = (1-(model_memory/total_memory)), includeNan=False, excludeID=[], excludeUUID=[])
        for i in range(len(tempID)):
            if len(tempID) > 0 and (tempID[i]==gpu_id):
                print("Found a gpu")
                print('We will use the GPU:',tempID[i],torch.cuda.get_device_name(tempID[i]))
                deviceID=[tempID[i]]
                return deviceID
            else:
                time.sleep(5)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def predict(params, device):
    with open("../../Gab_Data/gab_hate.json", "r") as fin:
        Gab_keyword_match= json.load(fin)
    
    tokenizer_emotions = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original",use_fast=False, cache_dir=params['cache_path'])
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False, cache_dir=params['cache_path'])
    model_emotion = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original",cache_dir=params['cache_path']).to(device)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",cache_dir=params['cache_path']).to(device)
    
    
    model_rationale.eval()
    model_emotion.eval()
    
    #Gab_keyword_match_prob=[]
    for i in tqdm(range(0,len(Gab_keyword_match),10000)):
        temp= Gab_keyword_match[i:i+10000]
        test_data_source = Prediction_Dataset(temp,tokenizer_emotions, params)
        test_dataloader= test_data_source.DataLoader
        predictions_emotion=[]
        for batch in test_dataloader:
          # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
              # Forward pass, calculate logit predictions
                outputs_emotion = model_emotion(b_input_ids,attention_mask=b_input_mask)
                
            
            logits_emotion = outputs_emotion[0]
            logits_emotion = torch.sigmoid(logits_emotion)
            # Move logits and labels to CPU
            logits_emotion = list(logits_emotion.detach().cpu().numpy())
            
            predictions_emotion+=logits_emotion
           
        
        test_data_source = Prediction_Dataset(temp,tokenizer_rationales, params)
        test_dataloader= test_data_source.DataLoader
        tokenized_sentences=test_data_source.tokenized_inputs
        predictions_rationale=[]
        for batch in test_dataloader:
          # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
              # Forward pass, calculate logit predictions
                _,outputs_rationale = model_rationale(b_input_ids,attention_mask=b_input_mask)
                
            
            logits_rationale = outputs_rationale
            logits_rationale = torch.softmax(logits_rationale.T, dim=0).T
            # Move logits and labels to CPU
            logits_rationale = list(logits_rationale.detach().cpu().numpy())
            
            predictions_rationale+=logits_rationale
            
        
#         print(len(predictions_emotion))
#         print(len(predictions_rationale))
        
#         print(predictions_rationale[0].shape)
#         print(len(tokenized_sentences[0]))
        
        
        for k in range(len(temp)):
            rationale_dict={}
            for x,y in zip(tokenized_sentences[k],predictions_rationale[k][1:]):
                rationale_dict[x]=y[1]
            
            emotion_dict={}   
            predictions_emotion[k]
            for idx,value in enumerate(predictions_emotion[k]):
                emotion_dict[model_emotion.config.id2label[idx]]=value
            temp[k]['emotion_dict']=emotion_dict
            temp[k]['rationale_dict']=rationale_dict
        
        with open('../../Gab_Data/features/gab_fear_hate_features'+str(i)+'.pickle', 'wb') as fout:
             pickle.dump(temp, fout,protocol=pickle.HIGHEST_PROTOCOL)    
 

        #Gab_keyword_match_prob+=temp

    
        

def predict_original(params, device):
    with open('dataset/final_dataset.json', 'r') as fp:
        json_data=json.load(fp)
    
    #emotion_bert='bhadresh-savani/bert-base-uncased-emotion'
    emotion_bert='monologg/bert-base-cased-goemotions-original'
    tokenizer_emotions = AutoTokenizer.from_pretrained(emotion_bert,use_fast=False, cache_dir=params['cache_path'])
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False, cache_dir=params['cache_path'])
    model_emotion = BertForMultiLabelClassification.from_pretrained(emotion_bert,cache_dir=params['cache_path']).to(device)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",cache_dir=params['cache_path']).to(device)
    
    
    model_rationale.eval()
    model_emotion.eval()
    
    json_data_modified=[]
    test_data_source = Org_Prediction_Dataset(json_data,tokenizer_emotions, params)
    test_dataloader = test_data_source.DataLoader
    predictions_emotion=[]
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
      # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            outputs_emotion = model_emotion(b_input_ids,attention_mask=b_input_mask)


        logits_emotion = outputs_emotion[0]
        logits_emotion = torch.sigmoid(logits_emotion)
        # Move logits and labels to CPU
        logits_emotion = list(logits_emotion.detach().cpu().numpy())

        predictions_emotion+=logits_emotion


    test_data_source = Org_Prediction_Dataset(json_data,tokenizer_rationales, params)
    test_dataloader= test_data_source.DataLoader
    tokenized_sentences=test_data_source.tokenized_inputs
    predictions_rationale=[]
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
      # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            _,outputs_rationale = model_rationale(b_input_ids,attention_mask=b_input_mask)


        logits_rationale = outputs_rationale
        logits_rationale = torch.softmax(logits_rationale.T, dim=0).T
        # Move logits and labels to CPU
        logits_rationale = list(logits_rationale.detach().cpu().numpy())

        predictions_rationale+=logits_rationale


#         print(len(predictions_emotion))
#         print(len(predictions_rationale))

#         print(predictions_rationale[0].shape)
#         print(len(tokenized_sentences[0]))

    k=0
    for key in json_data.keys():
        rationale_dict={}
        for x,y in zip(tokenized_sentences[k],predictions_rationale[k][1:]):
            rationale_dict[x]=str(y[1])

        emotion_dict={}   
        predictions_emotion[k]
        for idx,value in enumerate(predictions_emotion[k]):
            emotion_dict[model_emotion.config.id2label[idx]]=str(value)
        json_data[key]['emotion_dict']=emotion_dict
        json_data[key]['rationale_dict']=rationale_dict
        k+=1
        
    with open('dataset/final_dataset_emotion_rationale.json', 'w') as fp:
        json.dump(json_data, fp, indent=4)
    
    
    
    
params={
  'model':'bert',
  'features':'tfidf',
  'cache_path':'../../Saved_models/',
  'model_path':'Saved_Models/bert-base-uncased_fear_hate',
  #'model_path':'Hate-speech-CNERG/dehatebert-mono-english',
  'num_classes':3,
  'batch_size':64,
  'max_length':256,
  'learning_rate':2e-5 ,  ### learning rate 2e-5 for bert 0.001 for gru
  'epsilon':1e-8,
  'epochs':10,
  'dropout':0.1,
  'random_seed':2021,
  'device':'cuda',
  'save_path':'Saved_Models/',
  'logging':'neptune'
}
    


if __name__ == "__main__":
    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
#             deviceID = get_gpu(1)
#             torch.cuda.set_device(deviceID[0])
            #### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(1)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
    #fix_the_random(seed_val = params['random_seed'])
    
    
    predict(params, device)
    
    
