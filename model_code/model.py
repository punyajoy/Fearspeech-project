from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel

import torch.nn as nn
import torch


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






















#multilabel bert
class Bert_Multilabel(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None):
        outputs = self.bert(input_ids, mask)
        
        # out = outputs.last_hidden_state
        
        pooled_output = outputs[1]
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output

        


        



#multilabel roberta        
class Roberta_Multilabel(RobertaPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None):
        outputs = self.roberta(input_ids, mask)
        
        
        # out = outputs.last_hidden_state
        
        pooled_output = outputs[1]
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
        
        
#multilabel bert with rationale labels
class Bert_Multilabel_Rationale(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, rationale_vector=None):
        outputs = self.bert(input_ids, mask)
        
        # out = outputs.last_hidden_state
        last_hidden_states = outputs.last_hidden_state
        
        try:
            pooled_output = torch.ones(outputs[1].size()).to(rationale_vector.get_device())
        except:
            pooled_output = torch.ones(outputs[1].size())
        for i in range(0,last_hidden_states.size()[0]):
            pooled_output[i] = torch.matmul(rationale_vector[i], last_hidden_states[i])
            
        # print("SIZES: ",torch.matmul(rationale_vector[0], last_hidden_states[0]).size())
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
        
        
#multilabel roberta with rationale labels      
class Roberta_Multilabel_Rationale(RobertaPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, rationale_vector=None):
        outputs = self.roberta(input_ids, mask)
        
        
        # out = outputs.last_hidden_state
        last_hidden_states = outputs.last_hidden_state
        
        try:
            pooled_output = torch.ones(outputs[1].size()).to(rationale_vector.get_device())
        except:
            pooled_output = torch.ones(outputs[1].size())
        for i in range(0,last_hidden_states.size()[0]):
            pooled_output[i] = torch.matmul(rationale_vector[i], last_hidden_states[i])
            
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
        
#multilabel bert with emotions labels
class Bert_Multilabel_Emotions(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.num_emotions = 28
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.hidden_size = config.hidden_size
        self.classifier_new= nn.Linear(config.hidden_size + self.num_emotions, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, emotions_vector=None):
        outputs = self.bert(input_ids, mask)
        
        # out = outputs.last_hidden_state
        
        pooled_output = torch.cat((outputs[1],emotions_vector),dim=1)
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
               
        if(loss_label is not None):
            return output, loss_label
        else:
            return output

#multilabel roberta with emotions labels        
class Roberta_Multilabel_Emotions(RobertaPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.num_emotions = 28
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size + self.num_emotions, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, emotions_vector=None):
        outputs = self.roberta(input_ids, mask)
        
        
        # out = outputs.last_hidden_state
        
        pooled_output = torch.cat((outputs[1],emotions_vector),dim=1)
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output


#multilabel bert with rationale labels
class Bert_Multilabel_Rationale_Emotions(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.num_emotions = 28
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size + self.num_emotions, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, rationale_vector=None, emotions_vector=None):
        outputs = self.bert(input_ids, mask)
        
        # out = outputs.last_hidden_state
        last_hidden_states = outputs.last_hidden_state
        
        try:
            pooled_output = torch.ones(outputs[1].size()).to(rationale_vector.get_device())
        except:
            pooled_output = torch.ones(outputs[1].size())
        for i in range(0,last_hidden_states.size()[0]):
            pooled_output[i] = torch.matmul(rationale_vector[i], last_hidden_states[i])
        
        pooled_output = torch.cat((pooled_output,emotions_vector),dim=1)
            
        # print("SIZES: ",torch.matmul(rationale_vector[0], last_hidden_states[0]).size())
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output
        
        
#multilabel roberta with rationale labels      
class Roberta_Multilabel_Rationale_Emotions(RobertaPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.num_emotions = 28
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier_new= nn.Linear(config.hidden_size + self.num_emotions, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, labels=None, rationale_vector=None):
        outputs = self.roberta(input_ids, mask)
        
        
        # out = outputs.last_hidden_state
        last_hidden_states = outputs.last_hidden_state
        
        try:
            pooled_output = torch.ones(outputs[1].size()).to(rationale_vector.get_device())
        except:
            pooled_output = torch.ones(outputs[1].size())
        for i in range(0,last_hidden_states.size()[0]):
            pooled_output[i] = torch.matmul(rationale_vector[i], last_hidden_states[i])
            
        pooled_output = torch.cat((pooled_output[last_hidden_states.size()[0]-1],emotions_vector),dim=1)
        y_pred = self.classifier_new(self.dropout(pooled_output))
        output = torch.sigmoid(y_pred)
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.BCELoss()
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return output, loss_label
        else:
            return output