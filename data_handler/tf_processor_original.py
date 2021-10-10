from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import re
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
import random

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])



class Normal_Dataset():
    def __init__(self, data, tokenizer=None, params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.attn, self.labels, self.sentences = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    

    def tokenize(self, sentences):
        input_ids, attention_masks = [], []
        for sent in sentences:
            inputs=self.tokenizer.encode(sent,add_special_tokens=True,
                                              truncation=True,
                                              max_length=(self.max_length))
            
            input_ids.append(inputs)
            attention_masks.append([1]*len(inputs))
        return input_ids,attention_masks
    
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        count_error=0
        for label_list, sentence in tqdm(zip(list(data['majority_label']),list(data['text'])),total=len(data['majority_label'])):
            temp=[0,0,0]
            for label in label_list: 
                temp[self.label_dict[label]]=1
                
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                count_error+=1
                sentence = self.preprocess_func("dummy text")
            
            sentences.append(sentence)
            labels.append(list(temp))
            #print(sentence,label)
        print(random.sample(sentences, 5))
        print("No. of empty sequences", count_error)
        inputs, attn_mask = self.tokenize(sentences)
        return inputs, attn_mask, torch.Tensor(labels), sentences
    
    def get_attention_mask(self,attn_mask, maxlen=128):
        attn_mask_modified=[]
        for attn in attn_mask:
            attn = attn + [0]*(maxlen-len(attn))
            attn_mask_modified.append(attn)
        return attn_mask_modified
                                   
    def get_dataloader(self, inputs, attn_mask, labels, train = True):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_mask= self.get_attention_mask(attn_mask, maxlen=int(self.params['max_length']))
                                   
                                   
        input_ids=torch.tensor(inputs)
        attention_mask=torch.tensor(attention_mask)
        data = TensorDataset(input_ids,attention_mask,labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

    
    
class Modified_Dataset(Normal_Dataset):
    def __init__(self, data, tokenizer=None,  target_dict=None, params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        if(self.params['use_targets']):
            self.target_dict=target_dict
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.dict_features= self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.dict_features,self.train)
        
    def tokenize(self, sentences):
        input_ids, attention_masks = [], []
        for sent in sentences:
            inputs=self.tokenizer.encode(sent,add_special_tokens=True,
                                              truncation=True,
                                              max_length=(self.max_length))
            
            input_ids.append(inputs)
            attention_masks.append([1]*len(inputs))
        return input_ids,attention_masks
    
    
    def process_data(self, data):
        dict_output={'inputs':[],'attn_mask':[],'labels':[],'sentences':[]}
        if('rationales' in self.params['features']):
            dict_output['rationales']=[]
        if('emotion' in self.params['features']):
            dict_output['emotion']=[]
        if(self.params['use_targets']):
            dict_output['targets']=[]
        
        count_error=0
        for index,row in data.iterrows():
            if(self.train==True):
                if(self.params['labels_agg'] =='majority'):
                    temp=[0,0,0]
                    for label in row['majority_label']: 
                        temp[self.label_dict[label]]=1
                elif(self.params['labels_agg'] =='softlabel'):
                    count = {'normal':0,'hatespeech':0,'fearspeech':0}
                    for anno in row['annotations']:
                            for label in anno['Class']:
                                try:
                                    count[label]+=1
                                except KeyError:
                                    print(label)
                    
                    temp=[0,0,0]
                    for label in count.keys(): 
                        temp[self.label_dict[label]]=count[label]/len(row['annotations'])  
                else:
                    print("Label aggregation method not found")
            else:
                temp=[0,0,0]
                for label in row['majority_label']: 
                    temp[self.label_dict[label]]=1
            
            
            
            try:
                sentence = self.preprocess_func(row['text'])
            except TypeError:
                count_error+=1
                sentence = self.preprocess_func("dummy text")
            
            if(self.params['use_targets']):
                count_targets={}
                temp_targets=[]
                for label in self.target_dict.keys():
                    count_targets[label]=0
                    temp_targets.append(0)
                    
                for annotator in row['annotations']:
                    flag=True
                    for target in annotator['Targets']:
                        if target not in count_targets.keys():
                            if flag:
                                count_targets['Others']+=1
                                flag=False
                        else:
                            count_targets[target]+=1

                for label in count_targets.keys():
                    if count_targets[label]>=2:
                        temp_targets[self.target_dict[label]]=1
                
                dict_output['targets'].append(list(temp_targets))
            
            
            
            if('rationales' in self.params['features']):        
                special_tokens = self.tokenizer.encode("", add_special_tokens=True, truncation=True, max_length=(self.max_length))

                inputs = []
                rationales = []
                inputs.append(special_tokens[0])
                rationales.append(0)

                for key in row['rationale_dict'].keys():
                    inputs.append(self.tokenizer.convert_tokens_to_ids(key))
                    rationales.append(float(row['rationale_dict'][key]))

                inputs.append(special_tokens[1])
                rationales.append(0)
                dict_output['rationales'].append(rationales)
                dict_output['inputs'].append(inputs)
                dict_output['attn_mask'].append([1]*len(inputs))
            dict_output['sentences'].append(sentence)
            dict_output['labels'].append(list(temp))
            
            
            if('emotion' in self.params['features']):
                emotions = []
                for key in row['emotion_dict'].keys():
                        emotions.append(float(row['emotion_dict'][key]))
                dict_output['emotion'].append(emotions)
            
            
        print(random.sample(dict_output['sentences'], 5))
        print("No. of empty sequences", count_error)
        
        if('rationales' not in self.params['features']):
            dict_output['inputs'],dict_output['attn_mask'] = self.tokenize(dict_output['sentences'])
        
        
        
        return dict_output
                                
    def get_dataloader(self, dict_features, train = True):
        tensor_list=[]
        for key in dict_features.keys():
            
            if (key == 'inputs'):
                inputs = pad_sequences(dict_features['inputs'],maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
                tensor_list.append(torch.tensor(inputs))
            if (key == 'attn_mask'):
                attention_mask= self.get_attention_mask(dict_features['attn_mask'], maxlen=int(self.params['max_length']))
                tensor_list.append(torch.Tensor(attention_mask))
            if (key == 'labels'):
                tensor_list.append(torch.FloatTensor(dict_features['labels']))
            
            if (key == 'rationales'):
                rationales= self.get_attention_mask(dict_features['rationales'], maxlen=int(self.params['max_length']))
                tensor_list.append(torch.Tensor(rationales))
            
            if (key == 'emotion'):
                tensor_list.append(torch.Tensor(dict_features['emotion']))
            
            if (key == 'targets'):
                tensor_list.append(torch.Tensor(dict_features['targets']))
            
            
                
    
                                   
        
        
        data = TensorDataset(*tensor_list)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)


    
#Predictions class
class Prediction_Dataset():
    def __init__(self, data, tokenizer=None,  params=None):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        self.max_length=params['max_length']        
        self.tokenizer = tokenizer
        self.inputs, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    
    def tokenize(self, sentences):
        input_ids, attention_masks = [], []
        for sent in sentences:
            inputs=self.tokenizer.encode(sent,add_special_tokens=True,
                                              truncation=True,
                                              max_length=(self.max_length))
            
            input_ids.append(inputs)
            attention_masks.append([1]*len(inputs))
        return input_ids,attention_masks
    
    
    def process_data(self, data):
        sentences, attn = [], []
        count_error=0
        for element in data:
            try:
                sentence = self.preprocess_func(element['body'])
            except TypeError:
                count_error+=1
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            #print(sentence,label)
        print("No. of empty sequences", count_error)
        inputs, attn_mask = self.tokenize(sentences)
        return inputs, attn_mask
    
    
    def get_attention_mask(self,attn_mask, maxlen=128):
        attn_mask_modified=[]
        for attn in attn_mask:
            attn = attn + [0]*(maxlen-len(attn))
            attn_mask_modified.append(attn)
        return attn_mask_modified
                                   
    def get_dataloader(self, inputs, attn_mask):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_mask= self.get_attention_mask(attn_mask, maxlen=int(self.params['max_length']))
                                   
                                   
        input_ids=torch.tensor(inputs)
        attention_mask=torch.tensor(attention_mask)
        data = TensorDataset(input_ids,attention_mask)
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
 
    
#rationale labels
class Rationales_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.attn, self.labels, self.sentences, self.rationales = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels, self.rationales)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    

    def tokenize(self, data):
        input_ids, attention_masks, rationale_vectors = [], [], []
        for i in range(0,len(data)):
            special_tokens = self.tokenizer.encode("", add_special_tokens=True, truncation=True, max_length=(self.max_length))
            
            inputs = []
            rationales = []
            inputs.append(special_tokens[0])
            rationales.append(0)
            
            for key in data['rationale_dict'][i].keys():
                inputs.append(self.tokenizer.convert_tokens_to_ids(key))
                rationales.append(float(data['rationale_dict'][i][key]))
                
            inputs.append(special_tokens[1])
            rationales.append(0)
 
            input_ids.append(inputs)
            attention_masks.append([1]*len(inputs))
            rationale_vectors.append(rationales)
        return input_ids, attention_masks, rationale_vectors
    
    
    def process_data(self, data):
        sentences, labels = [], []
        count_error=0
        for label_list, sentence in tqdm(zip(list(data['majority_label']),list(data['text'])),total=len(data['majority_label'])):
            temp=[0,0,0]
            for label in label_list: 
                temp[self.label_dict[label]]=1
                
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                count_error+=1
                sentence = self.preprocess_func("dummy text")
            
            sentences.append(sentence)
            labels.append(list(temp))

        print(random.sample(sentences, 5))
        print("No. of empty sequences", count_error)
        inputs, attn_mask, rationales = self.tokenize(data)

        return inputs, attn_mask, torch.Tensor(labels), sentences, rationales
    
    def do_padding(self, req_vector, maxlen=128):
        modified=[]
        for vec in req_vector:
            vec = vec + [0]*(maxlen-len(vec))
            modified.append(vec)
        return modified
                                   
    def get_dataloader(self, inputs, attn_mask, labels, rationales, train = True):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_mask = self.do_padding(attn_mask, maxlen=int(self.params['max_length']))
        rationales = self.do_padding(rationales, maxlen=int(self.params['max_length']))
                                   
                                   
        input_ids=torch.tensor(inputs)
        attention_mask=torch.tensor(attention_mask)
        rationale_vectors=torch.tensor(rationales)
        data = TensorDataset(input_ids,attention_mask,labels,rationale_vectors)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
    
