#Use this python file to generate BERT embedding from csv files

from bert_serving.client import BertClient
import pandas as pd
import numpy as np


bc = BertClient()
df = pd.read_csv('./small_shoes.csv')
train_df = df[:25000]
train_df['rating'].value_counts()
reviews = [str(x).replace('\n', ' ') for x in train_df['review content'].values.tolist()]
len(reviews)
train_vec = bc.encode(reviews)
train_list = train_vec.tolist()
len(train_list)

train_vec = pd.DataFrame({'bert': train_list})
train_vec.head()

new_train_df = pd.concat([train_vec.reset_index(drop=True), train_df.reset_index(drop=True)], axis=1)
new_train_df.head()

new_train_df.to_csv('./dataset/shoes_bert_train.csv')