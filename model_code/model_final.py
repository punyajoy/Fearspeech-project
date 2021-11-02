from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel
from  transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, StableDropout, ContextPooler,DebertaModel
import torch.nn as nn
import torch


#rom github  https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca
class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

class Bert_Seq_Class(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.params=params
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.softmax=nn.Softmax(dim=1)
        self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None,attention_mask=None,labels=None):
        #batch[0] is input ids
        #batch[1] is attention_mask
        #batch[2] is labels
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        loss_label = None
        
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output



class Bert_Multilabel_Combined(BertPreTrainedModel):
    def __init__(self,config,params,weights):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.params=params
        self.num_emotion=params['emotion_num']
        self.num_targets=params['targets_num']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.weights=weights
        self.softmax=MaskedSoftmax()
            
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
            rationales=self.softmax(rationales,attention_mask)
            tensor1 = last_hidden_states.transpose(1,2)
            tensor2 = rationales.unsqueeze(2)
            pooled_output=torch.matmul(tensor1, tensor2).squeeze(2)
#             pooled_output = torch.ones(outputs[1].size()).to(rationales.get_device())
            
#             for i in range(0,last_hidden_states.size()[0]):
#                 pooled_output[i] = torch.matmul(rationales[i], last_hidden_states[i])
        
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
            loss_funct = nn.BCELoss(weight=self.weights)
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
        
        if(self.params['use_targets'] and (targets is not None)):
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_label += (0.5)*loss_targets

               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output

        
        
        
        
        
        
class Bert_Multilabel_Combined_Anno(BertPreTrainedModel):
    def __init__(self,config,params,weights):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.params=params
        self.num_emotion=params['emotion_num']
        self.num_targets=params['targets_num']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.weights=weights
        self.softmax=MaskedSoftmax()
            
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
        
        print(labels.shape)
        
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
            rationales=self.softmax(rationales,attention_mask)
            tensor1 = last_hidden_states.transpose(1,2)
            tensor2 = rationales.unsqueeze(2)
            pooled_output=torch.matmul(tensor1, tensor2).squeeze(2)
#             pooled_output = torch.ones(outputs[1].size()).to(rationales.get_device())
            
#             for i in range(0,last_hidden_states.size()[0]):
#                 pooled_output[i] = torch.matmul(rationales[i], last_hidden_states[i])
        
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
            loss_funct = nn.BCELoss(weight=self.weights)
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
        
        if(self.params['use_targets'] and (targets is not None)):
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_label += (0.5)*loss_targets

               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class Roberta_Multilabel_Combined(RobertaPreTrainedModel):
    def __init__(self,config,params,weights):
        super().__init__(config)
        
        self.num_labels=params['num_classes']
        self.params=params
        self.num_emotion=params['emotion_num']
        self.num_targets=params['targets_num']
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.weights=weights
        self.softmax=nn.Softmax(dim=1)

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
        
        outputs = self.roberta(input_ids, attention_mask)
        
        # out = outputs.last_hidden_state
        if('rationales' in self.params['features']):
            last_hidden_states = outputs.last_hidden_state
            rationales=self.softmax(rationales)
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
            loss_funct = nn.BCELoss(weight=self.weights)
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
        
        if(self.params['use_targets'] and (targets is not None)):
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_label += (0.5)*loss_targets

               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output

        
        
class Deberta_Multilabel_Combined(DebertaPreTrainedModel):
    def __init__(self, config, params,weights):
        super().__init__(config)
        self.params=params
        self.num_labels=params['num_classes']
        self.num_emotion=params['emotion_num']
        self.num_targets=params['targets_num']
        self.weights=weights
        self.softmax=nn.Softmax(dim=1)
        
        
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        
        if('emotion' in self.params['features']):
            self.classifier_new= nn.Linear(output_dim+self.num_emotion, self.num_labels)
        else:
            self.classifier_new= nn.Linear(output_dim, self.num_labels)
        self.dropout = StableDropout(params['dropout'])
        
            
            
        if(self.params['use_targets']):
            if('emotion' in self.params['features']):
                self.classifier_target= nn.Linear(output_dim+self.num_emotion, self.num_targets)
            else:
                self.classifier_target= nn.Linear(output_dim, self.num_targets)

        self.init_weights()

    def forward(self,input_ids=None,attention_mask=None,labels=None,
                    rationales=None,emotion_vector=None,targets=None):
        
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #batch[0] is input ids
        #batch[1] is attention_mask
        #batch[2] is labels
        #batch[3] is emotion vectors
        #batch[4] is rationale vectors
        #batch[5] is target labels
        outputs = self.deberta(input_ids, attention_mask)
        if('rationales' in self.params['features']):
            last_hidden_states = outputs.last_hidden_state
            rationales=self.softmax(rationales)
            tensor1 = last_hidden_states.transpose(1,2)
            tensor2 = rationales.unsqueeze(2)
            pooled_output=torch.matmul(tensor1, tensor2).squeeze(2)
        else:
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            
        if('emotion' in self.params['features']):
            pooled_output = torch.cat((pooled_output,emotion_vector),dim=1)
        
        
        
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        if(self.params['use_targets']):
            y_pred_target = self.classifier_target(self.dropout(pooled_output))
            output_target = torch.sigmoid(y_pred_target)


        
        loss_label = None
        
        if labels is not None:
            loss_funct = nn.BCELoss(weight=self.weights)
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
        
        if(self.params['use_targets'] and (targets is not None)):
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output_target.view(-1, self.num_targets), targets.view(-1, self.num_targets))
            loss_targets= loss_logits
            loss_label += (0.5)*loss_targets

               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
