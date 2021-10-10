from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel

import torch.nn as nn
import torch



class Bert_Multilabel_Combined(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.params=params
        self.num_emotion=28
        self.num_targets=params['targets_num']
        print(self.num_targets)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        if('emotion' in self.params['features']):
            self.classifier_new= nn.Linear(config.hidden_size+self.num_emotion, self.num_labels)
        else:
            self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
            
        if(self.params['use_targets']):
            if('emotion' in self.params['features']):
                self.classifier_target= nn.Linear(config.hidden_size+self.num_emotion, self.num_targets)
            else:
                self.classifier_target= nn.Linear(config.hidden_size, self.num_targets)

        
        self.init_weights()
        
    def forward(self, input_ids=None,attention_mask=None,labels=None,
                    rationales=None,emotion_vector=None,targets=None):
        #batch[0] is input ids
        #batch[1] is attention_mask
        #batch[2] is labels
        #batch[3] is emotion vectors
        #batch[4] is rationale vectors
        #batch[5] is target labels
        
        
        outputs = self.bert(input_ids, attention_mask)
        
        # out = outputs.last_hidden_state
        if('rationales' in self.params['features']):
            last_hidden_states = outputs.last_hidden_state
            tensor1 = last_hidden_states.transpose(1,2)
            tensor2 = rationales.unsqueeze(2)
            pooled_output=torch.matmul(tensor1, tensor2).squeeze(2)
        else:
            pooled_output = outputs[1]
        if('emotion' in self.params['features']):
            pooled_output = torch.cat((pooled_output,emotion_vector),dim=1)
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        if(self.params['use_targets']):
            y_pred_target = self.classifier_target(self.dropout(pooled_output))
            output_target = torch.sigmoid(y_pred_target)

        
        
        loss_label = None
        
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
        
        if(self.params['use_targets'] and (targets is not None)):
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_label= loss_logits + (0.5)*loss_targets

               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
