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
    def __init__(self, data, tokenizer=None,  params=None, train = False):
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
        if(self.params['label_type'] == 'soft'):
            for label_list, sentence in tqdm(zip(list(data['soft_labels']),list(data['text'])),total=len(data['soft_labels'])):
                temp=label_list
                
                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append((temp))
                #print(sentence,label)
            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask = self.tokenize(sentences)
            return inputs, attn_mask, torch.Tensor(labels), sentences
        
        else:
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



    
    
    
# class Modified_Dataset(Normal_Dataset):
#     def __init__(self, data, tokenizer=None,  params=None,train = False):
#         self.data = data
#         self.params= params
#         self.batch_size = self.params['batch_size']
#         self.train = train
#         self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
#         self.max_length=params['max_length']        
#         self.count_dic = {}
#         self.tokenizer = tokenizer
#         self.dict_features= self.process_data(self.data)
#         self.DataLoader = self.get_dataloader(self.dict_features,self.train)
        
#     def tokenize(self, sentences):
#         input_ids, attention_masks = [], []
#         for sent in sentences:
#             inputs=self.tokenizer.encode(sent,add_special_tokens=True,
#                                               truncation=True,
#                                               max_length=(self.max_length))
            
#             input_ids.append(inputs)
#             attention_masks.append([1]*len(inputs))
#         return input_ids,attention_masks
    
    
#     def process_data(self, data):
#         dict_output={'input_ids':[],'attn_mask':[],'labels':[],'sentences':[]}
#         if('rationales' in self.params['features']):
#             dict_output['rationales']=[]
#         if('emotion' in self.params['features']):
#             dict_output['emotion']=[]
        
#         count_error=0
#         for index,row in data.iterrows():
#             if(self.train==True):
#                 if(self.params['labels_agg'] =='majority'):
#                     temp=[0,0,0]
#                     for label in row['majority_label']: 
#                         temp[self.label_dict[label]]=1
#                 elif(self.params['labels_agg'] =='softlabel'):
#                     count = {'normal':0,'hatespeech':0,'fearspeech':0}
#                     for anno in row['annotations']:
#                             for label in anno['Class']:
#                                 count[label]+=1
                    
#                     temp=[0,0,0]
#                     for label in count.keys(): 
#                         temp[self.label_dict[label]]=count[label]/len(row['annotations'])                   
#             else:
#                 temp=[0,0,0]
#                 for label in row['majority_label']: 
#                     temp[self.label_dict[label]]=1
            
            
            
#             try:
#                 sentence = self.preprocess_func(row['text'])
#             except TypeError:
#                 count_error+=1
#                 sentence = self.preprocess_func("dummy text")
            
#             if('rationales' in self.params['features']):        
#                 special_tokens = self.tokenizer.encode("", add_special_tokens=True, truncation=True, max_length=(self.max_length))

#                 inputs = []
#                 rationales = []
#                 inputs.append(special_tokens[0])
#                 rationales.append(0)

#                 for key in row['rationale_dict'].keys():
#                     inputs.append(self.tokenizer.convert_tokens_to_ids(key))
#                     rationales.append(float(row['rationale_dict'][key]))

#                 inputs.append(special_tokens[1])
#                 rationales.append(0)

#             dict_output['inputs'].append(inputs)
#             dict_output['attn_mask'].append([1]*len(inputs))
#             dict_output['sentences'].append(sentence)
#             dict_output['labels'].append(list(temp))
            
            
#             if('emotion' in self.params['features']):
#                 emotions = []
#                 for key in emotion.keys():
#                         emotions.append(float(emotion[key]))
#                 dict_output['emotion'].append(emotions)
            
            
#         print(random.sample(sentences, 5))
#         print("No. of empty sequences", count_error)
        
#         if('rationales' in self.params['features']):
#             dict_output['inputs'],dict_output['attn_mask'] = self.tokenize(dict_output['sentences'])
        
#         return dict_output
                                
#     def get_dataloader(self, dict_features, train = True):
#         for key in dict_features:
            
        
#         inputs = pad_sequences(dict_features['inputs'],maxlen=int(self.params['max_length']), dtype="long", 
#                           value=self.tokenizer.pad_token_id, truncating="post", padding="post")
#         attention_mask= self.get_attention_mask(dict_output['attn_mask'], maxlen=int(self.params['max_length']))
#         rationales= self.get_attention_mask(dict_output['attn_mask'], maxlen=int(self.params['max_length']))
                                   
                                   
#         input_ids=torch.tensor(inputs)
#         attention_mask=torch.tensor(attention_mask)
#         labels=torch.tensor(labels)
        
        
#         if('emotion' in self.params['features']):
        
#         data = TensorDataset(input_ids,attention_mask,labels)
#         if self.train:
#             sampler = RandomSampler(data)
#         else:
#             sampler = SequentialSampler(data)
#         return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#Data handler for dataset with rationale labels
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
        if(self.params['label_type'] == 'soft'):
            for label_list, sentence in tqdm(zip(list(data['soft_labels']),list(data['text'])),total=len(data['soft_labels'])):
                temp=label_list
                
                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append((temp))
                #print(sentence,label)
            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask, rationales = self.tokenize(data)
            return inputs, attn_mask, torch.Tensor(labels), sentences, rationales
        
        else:
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

    
#Data handler for dataset with emotions labels
class Emotions_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.attn, self.labels, self.sentences, self.emotions_vector = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels, self.emotions_vector)
    
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
        sentences, labels, emotions_vector = [], [], []
        count_error=0
        if(self.params['label_type'] == 'soft'):
            for label_list, sentence, emotion in tqdm(zip(list(data['soft_labels']),list(data['text']),list(data['emotions'])),
                                                   total=len(data['soft_labels'])):
                temp=label_list
                emotions = []

                for key in emotion.keys():
                    emotions.append(float(emotion[key]))

                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append((temp))
                emotions_vector.append(emotions)

            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask = self.tokenize(sentences)

            return inputs, attn_mask, torch.Tensor(labels), sentences, torch.Tensor(emotions_vector)
        else:
            for label_list, sentence, emotion in tqdm(zip(list(data['majority_label']),list(data['text']),list(data['emotions'])),
                                                   total=len(data['majority_label'])):
                temp=[0,0,0]
                emotions = []
                for label in label_list: 
                    temp[self.label_dict[label]]=1

                for key in emotion.keys():
                    emotions.append(float(emotion[key]))

                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append((temp))
                emotions_vector.append(emotions)

            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask = self.tokenize(sentences)

            return inputs, attn_mask, torch.Tensor(labels), sentences, torch.Tensor(emotions_vector)
    
    def do_padding(self, req_vector, maxlen=128):
        modified=[]
        for vec in req_vector:
            vec = vec + [0]*(maxlen-len(vec))
            modified.append(vec)
        return modified
                                   
    def get_dataloader(self, inputs, attn_mask, labels, emotions_vector, train = True):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_mask = self.do_padding(attn_mask, maxlen=int(self.params['max_length']))
        #rationales = self.do_padding(rationales, maxlen=int(self.params['max_length']))
                                   
                                   
        input_ids=torch.tensor(inputs)
        attention_mask=torch.tensor(attention_mask)
        data = TensorDataset(input_ids,attention_mask,labels,emotions_vector)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)
    

#Data handler for dataset with rationale labels
class Rationales_Emotions_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict={'normal':0,'fearspeech':1,'hatespeech':2}
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.attn, self.labels, self.sentences, self.rationales, self.emotions_vector = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels, self.rationales, self.emotions_vector)
    
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
        sentences, labels, emotions_vector = [], [], []
        count_error=0
        if(self.params['label_type'] == 'soft'):
            for label_list, sentence, emotion in tqdm(zip(list(data['soft_labels']),list(data['text']),list(data['emotions'])),
                                                      total=len(data['soft_labels'])):
                
                temp=label_list
                emotions = []

                for key in emotion.keys():
                    emotions.append(float(emotion[key]))
                
                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append((temp))
                emotions_vector.append(emotions)

            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask, rationales = self.tokenize(data)
            return inputs, attn_mask, torch.Tensor(labels), sentences, rationales, torch.Tensor(emotions_vector)
        
        else:
            for label_list, sentence, emotion in tqdm(zip(list(data['majority_label']),list(data['text']),list(data['emotions'])),
                                             total=len(data['majority_label'])):
                temp=[0,0,0]
                for label in label_list:
                    temp[self.label_dict[label]]=1
                    
                emotions = []

                for key in emotion.keys():
                    emotions.append(float(emotion[key]))

                try:
                    sentence = self.preprocess_func(sentence)
                except TypeError:
                    count_error+=1
                    sentence = self.preprocess_func("dummy text")

                sentences.append(sentence)
                labels.append(list(temp))
                emotions_vector.append(emotions)
                #print(sentence,label)
            print(random.sample(sentences, 5))
            print("No. of empty sequences", count_error)
            inputs, attn_mask, rationales = self.tokenize(data)
            return inputs, attn_mask, torch.Tensor(labels), sentences, rationales, torch.Tensor(emotions_vector)
    
    def do_padding(self, req_vector, maxlen=128):
        modified=[]
        for vec in req_vector:
            vec = vec + [0]*(maxlen-len(vec))
            modified.append(vec)
        return modified
                                   
    def get_dataloader(self, inputs, attn_mask, labels, rationales, emotions_vector, train = True):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_mask = self.do_padding(attn_mask, maxlen=int(self.params['max_length']))
        rationales = self.do_padding(rationales, maxlen=int(self.params['max_length']))
                                   
                                   
        input_ids=torch.tensor(inputs)
        attention_mask=torch.tensor(attention_mask)
        rationale_vectors=torch.tensor(rationales)
        data = TensorDataset(input_ids,attention_mask,labels,rationale_vectors,emotions_vector)
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
        self.inputs, self.attn, self.tokenized_inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    
    def tokenize(self, sentences):
        tokenized_inputs, input_ids, attention_masks = [], [], []
        for sent in sentences:
            inputs=self.tokenizer.encode(sent,add_special_tokens=True,
                                              truncation=True,
                                              max_length=(self.max_length))
            input_tokens=self.tokenizer.tokenize(sent)
            tokenized_inputs.append(input_tokens)
            input_ids.append(inputs)
            attention_masks.append([1]*len(inputs))
        return input_ids,attention_masks,tokenized_inputs
    
    
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
        inputs, attn_mask, tokenized_inputs = self.tokenize(sentences)
        return inputs, attn_mask, tokenized_inputs
    
    
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
    
    
class Org_Prediction_Dataset(Prediction_Dataset):
    def process_data(self, data):
        sentences, attn = [], []
        count_error=0
        for element in data.values():
            try:
                sentence = self.preprocess_func(element['text'])
            except TypeError:
                count_error+=1
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            #print(sentence,label)
        print("No. of empty sequences", count_error)
        inputs, attn_mask, tokenized_inputs = self.tokenize(sentences)
        return inputs, attn_mask, tokenized_inputs
   
    
    
    
class Target_Dataset():
    def __init__(self, data, label_dict, tokenizer=None,  params=None, train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.label_dict=label_dict
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
        for i in tqdm(range(0,len(data))):
            count={}
            temp=[]
            for label in self.label_dict.keys():
                count[label]=0
                temp.append(0)
            for annotator in data['annotations'][i]:
                flag=True
                for target in annotator['Targets']:
                    if target not in count.keys():
                        if flag:
                            count['Others']+=1
                            flag=False
                    else:
                        count[target]+=1
            
            for label in count.keys():
                if count[label]>=2:
                    temp[self.label_dict[label]]=1
                    
            try:
                sentence = self.preprocess_func(data['text'][i])
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