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
import argparse
model_memory=10
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

            

def save_detection_model(model,tokenizer,params):
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir = params['save_path']+params['model_path']+'_'
    
    if(len(params['features'])!=0):
        output_dir+='_'.join(params['features'])
    if(params['use_targets']):
        output_dir+='_targets'
        
    output_dir+='_'+params['labels_agg']
    output_dir+='_external/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model=model.bert
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(params, os.path.join(output_dir, "training_args.bin"))

    
def save_json(dict1,type1,params):
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir ='Prediction_results/'+params['model_path']+'_'
    
    if(len(params['features'])!=0):
        output_dir+='_'.join(params['features'])
    if(params['use_targets']):
        output_dir+='_targets'
        
    output_dir+='_'+params['labels_agg']
    output_dir+='_external_'+type1+'.json'
    with open(output_dir, 'w') as outfile:
         json.dump(dict1, outfile, indent=2, cls=NumpyEncoder)
                
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_dataset(json_data):
    df=pd.DataFrame(json_data).transpose()
    return df


def evalphase(params,run,which_files='test',model=None,test_dataloader=None,device=None):
    print("Running eval on ",which_files,"...")
    logits_all=[]
    true_labels=[]
    pred_labels=[]
    t0 = time.time()
    model.eval()
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)


        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        batch = [element.to(device) for element in batch]
        input_ids=batch[0]
        attention_mask=batch[1]
        labels=batch[2]

        
        
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=None)
        logits = outputs
        
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        logits = logits.detach().cpu().numpy()
        label_ids = labels.detach().to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())
        logits_all+=list(logits)
    
    
    
    
    
#     pred_labels = np.array(pred_labels_raw) >= 0.5
#     true_labels = np.array([list(element) for element in true_labels])
    
    test_f1=f1_score(true_labels, pred_labels, average='macro')
    test_acc=accuracy_score(true_labels,pred_labels)
    test_precision=precision_score(true_labels, pred_labels, average='macro')
    test_recall=recall_score(true_labels, pred_labels, average='macro')
    test_hammingloss=hamming_loss(true_labels, pred_labels)
    
    
    key_dict={}
    key_dict['f1_macro']=test_f1
    key_dict['accuracy']=test_acc
    key_dict['precision']=test_precision
    key_dict['recall']=test_recall
    key_dict['hamming_loss']=test_hammingloss
    
    
    
    
    
    if(params['logging']=='neptune'):
        for key in key_dict:
            run[which_files+'/'+key].log(key_dict[key])
            
    
    
    
    
    # Report the final accuracy for this validation run.
    for key in key_dict:
        print("{0} : {1:.2f}".format(key, key_dict[key]))
    print(" Test took: {:}".format(format_time(time.time() - t0)))
    #print(ConfusionMatrix(true_labels,pred_labels))


    return key_dict, true_labels, pred_labels

def return_targets(dataset,threshold=20):
    label_dict={}
    for index,row in dataset.iterrows():
        for annotator in row['annotations']:
            for target in annotator['Targets']:
                try:    
                    label_dict[target]+=1
                except KeyError:
                    label_dict[target]=0
        
    target_dict = {}
    j=0
    for target in label_dict.keys():
        if label_dict[target] >= 20:
            target_dict[target]=j
            j+=1
    target_dict['Others']=j
    return target_dict



def train(params,run, device):
    if(run!=None):
        run["sys/tags"].add('baseline model')

    tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'])
        
    with open('../External_dataset/dynamic_hate/final_dataset_splits.json', 'r') as fp:
            post_id_dict=json.load(fp)
    with open('../External_dataset/dynamic_hate/final_dataset.json', 'r') as fp:
            json_data=json.load(fp)
    with open('../External_dataset/dynamic_hate/label_dict.json', 'r') as fp:
            label_dict=json.load(fp)
    
    
    
    dataset=pd.DataFrame(json_data).transpose()
    X_train=dataset[dataset['id'].isin(post_id_dict['train'])]
    X_val=dataset[dataset['id'].isin(post_id_dict['val'])]
    X_test=dataset[dataset['id'].isin(post_id_dict['test'])]
    print(len(X_train),len(X_val),len(X_test))
    print(X_train.columns)
    
   


    
    #### Need to change in case of other datasets
    train_data_source = External_Dataset(X_train,tokenizer,label_dict,params,train = True)
    val_data_source = External_Dataset(X_val,tokenizer,label_dict,params)
    test_data_source = External_Dataset(X_test,tokenizer,label_dict,params)
    
    
    train_dataloader= train_data_source.DataLoader
    val_dataloader= val_data_source.DataLoader
    test_dataloader= test_data_source.DataLoader
    if('deberta' in params['model_path']):
        model = DebertaV2_Multilabel_Combined.from_pretrained(
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
        #### For other type of models
        model = Bert_Seq_Class.from_pretrained(
            params['model_path'], # Use the 12-layer BERT model, with an uncased vocab.
            cache_dir=params['cache_path'],
            params=params).to(device)
        
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                )


    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params['epochs']

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),num_training_steps = total_steps)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    
    best_metrics_test={}
    best_metrics_val={}
    
    for key in ['f1_macro','precision','recall','accuracy','hamming_loss']:
        best_metrics_val[key]=0
        best_metrics_test[key]=0
        
    
    
    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention mask
            #   [2]: labels 
            
            batch = [element.to(device) for element in batch]
            input_ids=batch[0]
            attention_mask=batch[1]
            labels=batch[2]
            
#             rationales=None
#             emotions=None
#             targets=None
#             ind=2
#             if('rationales' in params['features']):
#                 ind+=1
#                 rationales=batch[ind]
#             if('emotion' in params['features']):
#                 ind+=1
#                 emotions=batch[ind]
#             if(params['use_targets']):
#                 ind+=1
#                 targets=batch[ind]
                
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            
            loss = outputs[1]
           
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            
                
            total_loss += loss.item()
            
            if(params['logging']=='neptune'):
                run['train/loss'].log(loss.item())
            
            
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        
        
        #train_dict,_,_=evalphase(params,run,'train',model,train_data_source_for_test.DataLoader,device)
        val_dict,val_true,val_pred=evalphase(params,run,'val',model,val_dataloader,device)
        test_dict,test_true,test_pred=evalphase(params,run,'test',model,test_dataloader,device)
        
        if(val_dict['f1_macro']>best_metrics_val['f1_macro']):
            for key in val_dict:
                best_metrics_val[key]=val_dict[key]
                best_metrics_test[key]=test_dict[key]
            dict_val={}
            for i in range(len(val_data_source.sentences)):
                dict_val[i]={}
                dict_val[i]['sentences']=val_data_source.sentences[i]
                dict_val[i]['true_labels']=val_true[i]
                dict_val[i]['pred_labels']=val_pred[i]
            
            dict_test={}
            for i in range(len(test_data_source.sentences)):
                dict_test[i]={}
                dict_test[i]['sentences']=test_data_source.sentences[i]
                dict_test[i]['true_labels']=test_true[i]
                dict_test[i]['pred_labels']=test_pred[i]
            
            
#             save_json(dict_test,'test',params)
#             save_json(dict_val,'val',params)   
            save_detection_model(model,tokenizer,params)
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print('avg_train_loss',avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
    if(run!=None):

        for key in best_metrics_val:
            run['val'+'/best_'+key]=best_metrics_val[key]
            run['test'+'/best_'+key]=best_metrics_test[key]

    
    
    del model
    torch.cuda.empty_cache()
    return 1




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
  'num_classes':2,
  'batch_size':16,
  'max_length':256,
  'learning_rate':5e-5 ,  ### learning rate 2e-5 for bert 0.001 for gru
  'epsilon':1e-8,
  'epochs':20,
  'dropout':0.1,
  'random_seed':2021,
  'device':'cuda',
  'use_targets':False,
  'targets_num':0,
  'features':[],
  'labels_agg':'majority',
  'save_path':'Saved_Models/',
  'logging':'local'
}

# features can be empty or contain 'emotion','rationales'
#'labels_agg' can be 'majority','crowd_layer','softlabel'


if __name__ == "__main__":
    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID = get_gpu(0)
            torch.cuda.set_device(deviceID[0])
            #### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
#             torch.cuda.set_device(1)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
        
    fix_the_random(seed_val = params['random_seed'])
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
        run["parameters"] = params
    else:
        pass
    
    train(params,run, device)