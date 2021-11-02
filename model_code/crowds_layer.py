'''
This script contains all the functionalities required to add the crowds layer.
'''

import torch
import pandas as pd
import numpy as np


class crowds_layer_dataset(torch.utils.data.Dataset):
    '''
    This is the Dataset Class for Crowds Classification Layer Masked MultiLabel Data.

    Attributes:
        annotations (pandas DataFrame): crowd sourced annotations.
        per_text_annotator_map (pandas DataFrame): annotation map per text per annotator.
        per_text_annotator_mask (pandas DataFrame): annotation mask per text per annotator(1 if participated, 0 otherwise).

    Methods:
        __len__ (returns: int): returns the length of the Dataset.
        __getitem__ (returns: triple): returns the data example present at the given index.
    '''
    def __init__(self, PATH_TO_ANNOTATIONS, PATH_TO_ANNOTATOR_MAP, PATH_TO_ANNOTATOR_MASK):
        '''
        This is the constructor.

        Parameters:
            PATH_TO_ANNOTATIONS (str): path to the complete annotations file(JSON).
            PATH_TO_ANNOTATOR_MAP (str): path to the complete annotation map file(JSON).
            PATH_TO_ANNOTATOR_MASK (str): path to the complete annotation mask(JSON).

        Returns:
            None
        '''
        self.annotations = pd.read_json(PATH_TO_ANNOTATIONS).transpose()
        self.per_text_annotator_map = pd.read_json(PATH_TO_ANNOTATOR_MAP)
        self.per_text_annotator_mask = pd.read_json(PATH_TO_ANNOTATOR_MASK)
        
    def __len__(self):
        '''
        This returns the length of the dataset.

        Parameters:
            None

        Returns:
            len(self.annotations) (int): total length of the dataset.
        '''
        return len(self.annotations)
        
    def __getitem__(self, index):
        '''
        This returns the data example present at the given index.

        Parameters:
            index (int): index at which the data example is to be retrieved.

        Returns:
            text, mapped_mask_arr, mapped_labels_arr (triple): returns the data example present at the given index.
        '''
        text = self.annotations.iloc[index, 0]
        mapped_labels = self.per_text_annotator_map.iloc[index, :]
        mapped_mask = self.per_text_annotator_mask.iloc[index, :]
        mapped_labels_arr = np.array(mapped_labels.tolist())
        mapped_mask_arr = np.array(mapped_mask.tolist()).reshape(-1, 1)
        
        return text, mapped_mask_arr, mapped_labels_arr
    

class crowds_classification_layer_masked_multilabel_loss(torch.nn.Module):
    '''
    This is a class for Crowds Classification Layer Masked MultiLabel Loss.

    Attributes:
        reduction (str): reduction type for the loss(set to 'none').
        sigmoid (torch.nn object): sigmoid layer.
        loss (torch.nn object): Binary Cross Entropy Loss(for Multiclass Multilabel Classification).

    Methods:
        forward (returns: tensor): returns the total masked multiclass multilabel loss for the input batch.
    '''
    def __init__(self):
        '''
        This is the constructor.

        Parameters:
            None

        Returns:
            None
        '''
        super().__init__()
        self.reduction = 'none'
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction = self.reduction)
        
    def forward(self, mask, y_pred, y_true):
        '''
        This performs the forward pass.

        Parameters:
            mask (tensor(torch.int): batch * num_annotators * 1): Mask per text per annotator(1 if participated else 0).
            y_pred (tensor(torch.float): batch * num_annotators * num_classes): Predictions(before the sigmoid layer).
            y_true (tensor(torch.int): batch * num_annotators * num_classes): True Labels. 

        Returns:
            final_loss (tensor(scalar)): cummulative loss for the input batch.
        '''
        # sigmoid
        y_pred = self.sigmoid(y_pred)
        y_true = y_true.to(torch.float)
        # loss
        loss = self.loss(y_pred, y_true)
        # mask
        final_loss = (loss * mask).sum()
        final_loss.requires_grad = True
        return final_loss


def get_batched_mapping(text_ids: list, df):
    '''
    Returns the per_annotator annotations for batched text input.

    Parameters:
        text_ids (list: str): IDs for the input texts.
        df (pandas DataFrame): Dataframe having the per_annotator mapping for all texts.

    Returns:
        Y (numpy 3D array): Per annotator labels for all texts. shape: num_texts * num_annotators * num_labels
    '''
    Y = None
    for i, text_id in enumerate(text_ids):
        curr = np.array(df.loc[text_id, :].tolist())
        curr_shape = curr.shape
        curr = curr.reshape((1, ) + curr_shape)
        
        if i == 0:
            Y = curr
        else:
            Y = np.concatenate((curr, Y), 0)
    return Y


class CrowdsClassificationLayer(torch.nn.Module): 
    '''
    This is a class for Crowds Classification Layer.

    Attributes:
        output_dim (int): Number of classes present for classification.
        num_annotators (int): Number of annotators present.
        conn_type (str): Type of mapping for the crowds layer.

    Methods:
        build (returns: None): initializes the weights and biases according to the conn_type.
        forward (returns: tensor): returns the output of the mapping function for given input.
        compute_output_shape (returns: triple): returns the shape of the output of forward for given input.
        init_identities (returns: tensor): returns the initialized weights tensor for conn_type: 'MW', 'MW+B'.
    '''

    def __init__(self, output_dim, num_annotators, conn_type="MW"):
        '''
        This is the constructor.

        Parameters:
            output_dim (int): Number of classes present for classification.
            num_annotators (int): Number of annotators present.
            conn_type (int): Type of mapping for the crowds layer. 

        Returns:
            None
        '''
        super(CrowdsClassificationLayer, self).__init__()
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.build()  # initialize weights and biases according to conn_type

    def build(self):
        '''
        Initializes the weights and biases according to the conn_type.

        Supported conn_type:
            'MW': matrix of weights per annotator.
            'VW': vector of weights (one scale per class) per annotator.
            'MW+B': one matrix of weights and one bias per class per annotator.
            'VW+B': two vectors of weights (one scale and one bias per class) per annotator.
            'SW': single weight value per annotator.

        Parameters:
            None

        Returns:
            None
        '''
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = torch.nn.Parameter(self.init_identities((self.num_annotators, self.output_dim, self.output_dim)))
        elif self.conn_type == "VW":
            # vector of weights (one scale per class) per annotator  
            self.kernel = torch.nn.Parameter(torch.ones(self.num_annotators, self.output_dim))
        elif self.conn_type == "MW+B":
            # one matrix of weights and one bias per class per annotator
            self.kernel = []
            self.kernel.append(torch.nn.Parameter(self.init_identities((self.num_annotators, self.output_dim, self.output_dim))))
            self.kernel.append(torch.nn.Parameter(torch.zeros(self.num_annotators, self.output_dim)))
        elif self.conn_type == "VW+B": 
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(torch.nn.Parameter(torch.ones(self.num_annotators, self.output_dim)))
            self.kernel.append(torch.nn.Parameter(torch.zeros(self.num_annotators, self.output_dim)))
        elif self.conn_type == "SW":
            # single weight value per annotator
            self.kernel = torch.nn.Parameter(torch.ones(self.num_annotators,1))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")


    def forward(self, x):
        '''
        Returns the output of the mapping function for given input.
        
        Parameters:
            x (tensor): input tensor with shape (batch_size, output_dim).

        Returns:
            res (tensor): mapped output for the input, having shape (batch_size , num_annotators, output_dim).
        '''
        res = None
        if self.conn_type == "MW":
            res = torch.matmul(x, self.kernel)
        elif self.conn_type == "VW":
            out = []
            for i in range(self.num_annotators):
                out.append(x*self.kernel[i])
                res = torch.stack(out)
        elif self.conn_type == "MW+B":
            res = torch.matmul(x, self.kernel[0])
            out = []
            for i in range(self.num_annotators):
                out.append(x*self.kernel[1][i]) 
            res += torch.stack(out)
        elif self.conn_type == "VW+B":
            out = []
            for i in range(self.num_annotators):
                out.append(x*self.kernel[0][i] + x*self.kernel[1][i]) 
            res = torch.stack(out)
        elif self.conn_type == "SW":
            out = []
            for i in range(self.num_annotators):
                out.append(x*self.kernel[i])
            res = torch.stack(out)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!") 
            
        res = torch.permute(res, [1, 0, 2])          
        return res

    def compute_output_shape(self, input_shape):
        '''
        Returns the shape of the output of forward for given input.
        
        Parameters:
            input_shape (triple): shape of the input to the crowds layer.

        Returns:
            output_shape (triple): shape of the output of the crowds layer.
        '''
        output_shape = (self.num_annotators,) + input_shape
        return output_shape

    def init_identities(self, shape):
        '''
        Returns the initialized weights tensor for conn_type: 'MW', 'MW+B'.

        Parameters:
            shape (triple): target shape of the returned tensor.

        Returns:
            out_ (tensor): initialized weights tensor for conn_type: 'MW', 'MW+B'.
        '''
        out = []
        for i in range(self.num_annotators):
            out.append(torch.eye(self.output_dim))
        out_ = torch.stack(out) 
        return out_