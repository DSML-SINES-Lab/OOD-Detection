import data_helpers as dh
import pandas as pd



def load_encoded_BPS(path):
  data = pd.read_csv(path,header = None)
  labeled_data = pd.get_dummies(data)
  data = data.iloc[:,:-1].values
  labeled_data = labeled_data.iloc[:,120:].values
  vocab,invVocab = dh.build_vocab(data[:,:])
  return data,labeled_data,vocab,invVocab








