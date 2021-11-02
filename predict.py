import pandas as pd
from glob import glob
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from data_handler.tf_processor_original import  *
from model_code.model_final import *
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

def load_detection_path(params):
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir = params['save_path']+params['model_path']+'_'
    
    if(len(params['features'])!=0):
        output_dir+='_'.join(params['features'])
    if(params['use_targets']):
        output_dir+='_targets'
        
    output_dir+='_'+params['labels_agg']
    output_dir+='_fear_hate/'
    # Create output directory if needed
    return output_dir
        
    
def predict(params, device):
    params['model_path']=load_detection_path(params)
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'])
    class_weights=torch.tensor([1,1,1]).to(device)
    
    if('deberta' in params['model_path']):
        model = Deberta_Multilabel_Combined.from_pretrained(
            params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
            cache_dir=params['cache_path'],
            params=params,
            weights=class_weights).to(device)
    elif(('roberta' in params['model_path']) or (params['model_path']=='vinai/bertweet-base')):
        model = Roberta_Multilabel_Combined.from_pretrained(
            params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
            cache_dir=params['cache_path'],
            params=params,
            weights=class_weights).to(device)
    else:
        model = Bert_Multilabel_Combined.from_pretrained(
            params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
            cache_dir=params['cache_path'],
            params=params,
            weights=class_weights).to(device)
      
    model.eval()
    
    
    files=glob('../../Gab_Data/features/*.pickle')
    for file in tqdm(files,total=len(files)):
        with open(file, 'rb') as handle:
            temp = pickle.load(handle)
        test_data_source = Prediction_Dataset(temp,tokenizer, params)
        test_dataloader= test_data_source.DataLoader
        predictions=[]
        for batch in test_dataloader:
          # Add batch to GPU
        
            batch = [element.to(device) for element in batch]
            input_ids=batch[0]
            attention_mask=batch[1]
            rationales=None
            emotions=None
            targets=None
            ind=1
            if('rationales' in params['features']):
                ind+=1
                rationales=batch[ind]
            if('emotion' in params['features']):
                ind+=1
                emotions=batch[ind]
                
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
              # Forward pass, calculate logit predictions
              outputs = model(input_ids=input_ids,attention_mask=attention_mask,
                            rationales=rationales,emotion_vector=emotions)


            logits = outputs

            # Move logits and labels to CPU
            logits = list(logits.detach().cpu().numpy())
            
            predictions+=logits
        
        print(len(predictions))
    
        logits_all_final=[]
        for logits in predictions:
            logits_all_final.append(logits)

        for k in range(len(temp)):
            temp[k]['predicted_probab']=logits_all_final[k]
        with open(file, 'wb') as fout:
             pickle.dump(temp, fout,protocol=pickle.HIGHEST_PROTOCOL)    
 
    
    
    

params={
  'model':'bert',
  'features':'tfidf',
  'cache_path':'../../Saved_models/',
  'model_path':'roberta-base',
  'num_classes':3,
  'batch_size':8,
  'max_length':256,
  'learning_rate':5e-5 ,  ### learning rate 2e-5 for bert 0.001 for gru
  'epsilon':1e-8,
  'epochs':20,
  'dropout':0.5,
  'random_seed':2021,
  'device':'cuda',
  'use_targets':True,
  'targets_num':22,
  'features':['emotion'],
  'labels_agg':'majority',
  'save_path':'Saved_Models/',
  'logging':'local'
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
    
    
    predict(params, device)
    
    
