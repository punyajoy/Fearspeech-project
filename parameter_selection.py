import pandas as pd
from glob import glob
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from data_handler.tf_processor_original import  *
# from data_handler.simple_processor import *
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
from sklearn.model_selection import ParameterGrid




#bert-base-uncased
#roberta-base
#Hate-speech-CNERG/dehatebert-mono-english
#Hate-speech-CNERG/bert-base-uncased-hatexplain
#Saved_Models/hate_bert
#Gab-dataset finetune 
#Implcit hate finetune
    
params={
  'model':'bert',
  'features':'tfidf',
  'cache_path':'../../Saved_models/',
  'model_path':'bert-base-uncased',
  #'model_path':'Hate-speech-CNERG/dehatebert-mono-english',
  'num_classes':3,
  'batch_size':16,
  'max_length':256,
  'learning_rate':5e-5 ,  ### learning rate 2e-5 for bert 0.001 for gru
  'epsilon':1e-8,
  'epochs':10,
  'dropout':0.1,
  'random_seed':2021,
  'device':'cuda',
  'use_targets':True,
  'targets_num':0,
  'emotion_num':6,
  'features':[],
  'labels_agg':'majority',
  'save_path':'Saved_Models/',
  'logging':'neptune'
}

# features can be empty or contain 
#'labels_agg' can be 'majority','crowd_layer','softlabel'


if __name__ == "__main__":
    params_list = []
    params_new = {}
    for key in params.keys():
        params_new[key]=[params[key]]
    
   
    params_new['model_path']=['bert-base-uncased','roberta-base','microsoft/deberta-base']
    params_new['learning_rate']=[2e-5, 2e-4]
    params_new['labels_agg']=['softlabel', 'majority']
    params_new['use_targets']=[True]
    params_new['features']=[['emotions'],['rationales','emotions'],[]]
    params_new['dropout']=[0.2, 0.5]
    
    params_list=list(ParameterGrid(params_new))
    print('Total experiments to be done:',len(params_list))
    
    with open('all_params.json', 'w') as fout:
            json.dump(params_list ,fout,indent=4)