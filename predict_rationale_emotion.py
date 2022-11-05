import pandas as pd
from glob import glob
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from model_code.rationale_model import *
from data_handler.tf_processor import  *
from model_code.model import *
from tqdm import tqdm
import zstandard as zstd
import ujson
import gensim
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
import pickle5 as pickle

model_memory=7
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
    
    tokenizer_emotions = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original",use_fast=False)
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False)
    model_emotion = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original").to(device)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two").to(device)
    
    
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
        
        with open('../../Gab_Data/new_features/gab_fear_hate_features'+str(i)+'.pickle', 'wb') as fout:
             pickle.dump(temp, fout,protocol=pickle.HIGHEST_PROTOCOL)    
 

        #Gab_keyword_match_prob+=temp


class Zreader:
    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
        self.chunk_size = chunk_size
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''


    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            chunk = self.reader.read(self.chunk_size).decode()
            if not chunk:
                break
            lines = (self.buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line

            self.buffer = lines[-1]
            
            
            
def predict_whole(params, device):
    file_gab='../../Gab_Data/gab_corpus2.ndjson.zst'
    zreader=Zreader(file_gab,chunk_size=2**16)
    tokenizer_emotions = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original",use_fast=False)
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False)
    model_emotion = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original").to(device)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two").to(device)
    
    model_rationale.eval()
    model_emotion.eval()

    
    pbar = tqdm(total=1000000)
    flag=0
    count=0
    files_index=0
    posts=[]
    while True:
        pbar.update(1)
        try:
            chunk1 = zreader.reader.read(zreader.chunk_size)
            chunk_decoded = chunk1.decode()
        except UnicodeDecodeError:
            chunk2 = zreader.reader.read(100)
            try:
                chunk_decoded = (chunk1+chunk2).decode()
            except UnicodeDecodeError:
                continue
        if not chunk_decoded:
            break
        lines = (zreader.buffer + chunk_decoded).split("\n")
        for line in lines[:-1]:
            try:
                temp = ujson.loads(line)
                posts.append(temp)
                count+=1
            except:
                pass
        
        ### prediction json
        if(count>100000):
            print(count,len(posts))
            files_index+=1
            print(files_index)
            try:
                with open('../../Gab_Data/new_features/gab_fear_hate_features'+str(files_index)+'.pkl', 'rb') as handle:
                     Gab_keyword_match = pickle.load(handle)

                if(params['reset']==False):
                    if('emotion_dict' in Gab_keyword_match[1].keys()):
                        posts=[]
                        count=0
                        continue
            except FileNotFoundError as e:
                pass
            
            test_data_source = Prediction_Dataset(posts,tokenizer_emotions, params)
            test_dataloader= test_data_source.DataLoader
            predictions_emotion=[]
            for batch in tqdm(test_dataloader,total=len(test_dataloader)):
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
            
            test_data_source = Prediction_Dataset(posts,tokenizer_rationales, params)
            test_dataloader= test_data_source.DataLoader
            tokenized_sentences=test_data_source.tokenized_inputs
            predictions_rationale=[]
            for batch in tqdm(test_dataloader,total=len(test_dataloader)):
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
            
            for k in range(len(posts)):
                rationale_dict={}
                for x,y in zip(tokenized_sentences[k],predictions_rationale[k][1:]):
                    rationale_dict[x]=y[1]

                emotion_dict={}   
                predictions_emotion[k]
                for idx,value in enumerate(predictions_emotion[k]):
                    emotion_dict[model_emotion.config.id2label[idx]]=value
                posts[k]['emotion_dict']=emotion_dict
                posts[k]['rationale_dict']=rationale_dict
            
            print(files_index)
            with open('../../Gab_Data/new_features/gab_fear_hate_features'+str(files_index)+'.pickle', 'wb') as fout:
                 pickle.dump(posts, fout,protocol=pickle.HIGHEST_PROTOCOL)    
            
            posts=[]
            count=0
        zreader.buffer = lines[-1]
    pbar.close() 
    print(count)


    
    
def predict_whole_emotions(params, device):
    tokenizer_emotions = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original",use_fast=False)
    model_emotion = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original").to(device)
    model_emotion.eval()

    
#     with open('/home/punyajoy/Gab_Data_old/Final_Posts.pkl','rb') as fp:
#         dict_posts = pickle.load(fp)
    
#     files=sorted(glob('../../Gab_Data/new_features_old_gab/*.pickle'))
    files=sorted(glob('../../../../Newhd/Punyajoy_folders/works_2021/Fearspeech_Additional/Facebook_Data/*.pkl'))
    
    for file in tqdm(files,total=len(files)):
        with open(file, 'rb') as handle:
            temp = pickle.load(handle)
            
        if(params['reset']==False):
            if('emotion_dict' in temp[1].keys()):
                 continue
        test_data_source = Prediction_Dataset(temp,tokenizer_emotions, params)
        test_dataloader= test_data_source.DataLoader
        predictions_emotion=[]
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
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

        for k in range(len(temp)):
            emotion_dict={}   
            predictions_emotion[k]
            for idx,value in enumerate(predictions_emotion[k]):
                emotion_dict[model_emotion.config.id2label[idx]]=value
            temp[k]['emotion_dict']=emotion_dict

        with open(file, 'wb') as fout:
             pickle.dump(temp, fout,protocol=pickle.HIGHEST_PROTOCOL)    

                
                
def predict_whole_rationales(params, device):
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two").to(device)
    
    model_rationale.eval()
    
    
#     with open('/home/punyajoy/Gab_Data_old/Final_Posts.pkl','rb') as fp:
#         dict_posts = pickle.load(fp)
    files=sorted(glob('../../../../Newhd/Punyajoy_folders/works_2021/Fearspeech_Additional/Facebook_Data/*.pkl'))
    
#     files=sorted(glob('../../Gab_Data/new_features_old_gab/*.pickle'))
    for file in tqdm(files,total=len(files)):
        with open(file, 'rb') as handle:
            temp = pickle.load(handle)
            
        if(params['reset']==False):
            if('rationale_dict' in temp[1].keys()):
                 continue
        test_data_source = Prediction_Dataset(temp,tokenizer_rationales, params)
        test_dataloader= test_data_source.DataLoader
        tokenized_sentences=test_data_source.dict_features['tokenized_inputs']
        predictions_rationale=[]
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
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

        for k in range(len(temp)):
            rationale_dict={}
            for x,y in zip(tokenized_sentences[k],predictions_rationale[k][1:]):
                rationale_dict[x]=y[1]

            temp[k]['rationale_dict']=rationale_dict

        with open(file, 'wb') as fout:
             pickle.dump(temp, fout,protocol=pickle.HIGHEST_PROTOCOL)    
                
                
                
                
                
                
                
                
                
   

def predict_original(params, device):
    with open('dataset/final_dataset.json', 'r') as fp:
        json_data=json.load(fp)
    
    #emotion_bert='bhadresh-savani/bert-base-uncased-emotion'
    emotion_bert='monologg/bert-base-cased-goemotions-original'
    tokenizer_emotions = AutoTokenizer.from_pretrained(emotion_bert,use_fast=False)
    tokenizer_rationales = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",use_fast=False)
    model_emotion = BertForMultiLabelClassification.from_pretrained(emotion_bert).to(device)
    model_rationale = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two").to(device)
    
    
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
  'batch_size':128,
  'reset':True,
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
            torch.cuda.set_device(0)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
    #fix_the_random(seed_val = params['random_seed'])
    
    predict_whole_rationales(params, device)
    predict_whole_emotions(params, device)
    
#     predict_original(params, device)
    
    
