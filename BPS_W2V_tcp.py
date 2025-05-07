import data_helpers as dh
import pandas as pd



def load_encoded_BPS(path):
  
  data = pd.read_csv(path,header = None)
  labeled_data = pd.get_dummies(data.iloc[:,-1])
    
  data = data.iloc[:,:-1].values
  unique_labels=labeled_data.columns
  labeled_data=labeled_data.values
  vocab,invVocab = dh.build_vocab(data[:,:])
  x, y = dh.build_input_data(data, labeled_data, vocab)

  return x,y,unique_labels,vocab,invVocab

def load_encoded_BPS2(path,oldvacb):
  data = pd.read_csv(path,header = None)
    
  labeled_data = pd.get_dummies(data.iloc[:,-1])
    
  data = data.iloc[:,:-1].values
  unique_labels=labeled_data.columns
  labeled_data=labeled_data.values
  vocab,invVocab = dh.update_vocab(data[:,:],oldvacb)
  x, y = dh.build_input_data(data, labeled_data, vocab)
  
  return x,y,unique_labels,vocab,invVocab






