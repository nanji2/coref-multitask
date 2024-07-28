

from datasets import load_dataset
import os
import csv
import nltk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
import pandas as pd
import re

from transformers import AutoTokenizer, AutoModel


os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask')
BERT_PATH = './bert-large-uncased' # the path of your downloaded pre-trained language model
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert = AutoModel.from_pretrained(BERT_PATH)

dataset_train = load_dataset("definite_pronoun_resolution",split='train')
dataset_test = load_dataset("definite_pronoun_resolution",split='test')

os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask\dpr')
dataset_train.to_csv("dpr_train.csv")
dataset_test.to_csv("dpr_test.csv")

dpr_train = pd.read_csv("dpr_train.csv")
dpr_test = pd.read_csv("dpr_test.csv")

# manually change some " to ' in the cnadidates column
dpr_train['candidates'] = dpr_train['candidates'].str.replace('\"','\'')
dpr_test['candidates'] = dpr_test['candidates'].str.replace('\"','\'')
#i=5

def return_new_dpr_df(dpr_df_original):
    # if want to insert list into cells, must set column type as object
    dpr_df=dpr_df_original.copy(deep=True)
    print('start')
    dpr_df['original_position_list']=""
    dpr_df['original_position_list']=dpr_df['original_position_list'].astype('object')
    dpr_df['pronoun_words_positions']=""
    dpr_df['pronoun_words_positions']=dpr_df['pronoun_words_positions'].astype('object')
    dpr_df['A_words_positions']=""
    dpr_df['A_words_positions']=dpr_df['A_words_positions'].astype('object')
    dpr_df['B_words_positions']=""
    dpr_df['B_words_positions']=dpr_df['B_words_positions'].astype('object')
    dpr_df['label1']=""
    dpr_df['label1']=dpr_df['label1'].astype('object')
    
    
    for i in range(dpr_df.shape[0]):
        print(i)
        test_sen=dpr_df.at[i,'sentence']
        roberta_idx=tokenizer.encode(test_sen)
        tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
        
        
        test_sen=dpr_df.at[i,'sentence']
        dpr_df.at[i,'sequence']=test_sen
        
        roberta_idx=tokenizer.encode(test_sen)
        # if the list is of integers, convert the elements before joining them
        dpr_df.at[i,'roberta_idx']=' '.join(str(e) for e in roberta_idx)
        
        tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
        dpr_df.at[i,'roberta_token']=' '.join(tokenized_testsen)
        
        
        sentence = test_sen
        b = []
        b.append(([101],))
        for m in re.finditer(r'\S+', sentence):
          w = m.group(0)
          t = (tokenizer.encode(w, add_special_tokens=False), (m.start(), m.end()-1))
          b.append(t)
        b.append(([102],))
        
        original_position_list=[]
        first_token_list=[0]*len(tokenized_testsen)
        original_position=0
        b_position=0
        counter=0
        
        for m in range(len(tokenized_testsen)):
            if tokenized_testsen[m]=='[CLS]':
                original_position_list.append(-1)
                b_position+=1
            elif tokenized_testsen[m]=='[SEP]':
                original_position_list.append(-1)
            else:
                same_word_token_id_list=b[b_position][0]
                if len(same_word_token_id_list)==1:
                    original_position=b_position-1
                    original_position_list.append(original_position)
                    b_position+=1
                    first_token_list[m]=1
                elif len(same_word_token_id_list)>1:
                    if counter<len(same_word_token_id_list):
                        if counter==0:
                            first_token_list[m]=1
                        original_position=b_position-1
                        original_position_list.append(original_position)
                        counter+=1
                        if counter>=len(same_word_token_id_list):
                            counter=0
                            b_position+=1
        
        
        
        dpr_df.at[i,'original_position_list']=original_position_list
        dpr_df.at[i,'first_token']=' '.join(str(e) for e in first_token_list)
        
        pronoun_offset=dpr_df.at[i,'sequence'].find(' '+dpr_df.at[i,'pronoun'])+1
        pronoun_start_word_position=test_sen[:pronoun_offset].count(' ')
        pronoun_num_of_spaces=dpr_df.at[i,'pronoun'].count(' ')
        pronoun_end_word_position=pronoun_start_word_position+pronoun_num_of_spaces
        pronoun_words_positions=[pronoun_start_word_position,pronoun_end_word_position]
        dpr_df.at[i,'pronoun_words_positions']=pronoun_words_positions
        #dpr_df.at[i,'pronoun_word_position']=test_sen[:pronoun_offset].count(' ')
        
        
        dpr_df.at[i,'candidates']=dpr_df.at[i,'candidates'][2:-2].split('\' \'')
        
        
        
        A_offset=dpr_df.at[i,'sequence'].find(dpr_df.at[i,'candidates'][0])
        A_start_word_position=test_sen[:A_offset].count(' ')
        A_num_of_spaces=dpr_df.at[i,'candidates'][0].count(' ')
        A_end_word_position=A_start_word_position+A_num_of_spaces
        A_words_positions=[A_start_word_position,A_end_word_position]
        dpr_df.at[i,'A_words_positions']=A_words_positions
        
        B_offset=dpr_df.at[i,'sequence'].find(dpr_df.at[i,'candidates'][1])
        B_start_word_position=test_sen[:B_offset].count(' ')
        B_num_of_spaces=dpr_df.at[i,'candidates'][1].count(' ')
        B_end_word_position=B_start_word_position+B_num_of_spaces
        B_words_positions=[B_start_word_position,B_end_word_position]
        dpr_df.at[i,'B_words_positions']=B_words_positions
        
        label1_list=[0]*len(tokenized_testsen)
        if dpr_df.at[i,'label']==0:
            for m in range(len(tokenized_testsen)):
                if dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'pronoun_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'pronoun_words_positions'][1]:
                    label1_list[m]=1
                elif dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'A_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'A_words_positions'][1]:
                    label1_list[m]=1
        elif dpr_df.at[i,'label']==1:
            for m in range(len(tokenized_testsen)):
                if dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'pronoun_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'pronoun_words_positions'][1]:
                    label1_list[m]=1
                elif dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'B_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'B_words_positions'][1]:
                    label1_list[m]=1
        
        dpr_df.at[i,'label1']=' '.join(str(e) for e in label1_list)
    
    # repeat each row for two times
    dpr_df=dpr_df.loc[dpr_df.index.repeat(2)].reset_index(drop=True)
    
    for i in range(dpr_df.shape[0]):
        
        # A-P case
        if i%2==0:
            if dpr_df.at[i,'label']==0:
                dpr_df.at[i,'label2']=1
            elif dpr_df.at[i,'label']==1:
                dpr_df.at[i,'label2']=0
            #dpr_df.at[i,'label2']=int(dpr_df.at[i,'A-coref'])
        # B-P case
        if i%2==1:
            if dpr_df.at[i,'label']==1:
                dpr_df.at[i,'label2']=1
            elif dpr_df.at[i,'label']==0:
                dpr_df.at[i,'label2']=0
                
        test_sen=dpr_df.at[i,'sentence']
        tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
        mask_ab=[0]*len(tokenized_testsen)
        mask_p=[0]*len(tokenized_testsen)
        # A-P case
        if i%2==0:
            for m in range(len(tokenized_testsen)):
                if dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'pronoun_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'pronoun_words_positions'][1]:
                    mask_p[m]=1
                elif dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'A_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'A_words_positions'][1]:
                    mask_ab[m]=1
        # B-P case
        elif i%2==1:
            for m in range(len(tokenized_testsen)):
                if dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'pronoun_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'pronoun_words_positions'][1]:
                    mask_p[m]=1
                elif dpr_df.at[i,'original_position_list'][m]>=dpr_df.at[i,'B_words_positions'][0] \
                and dpr_df.at[i,'original_position_list'][m]<=dpr_df.at[i,'B_words_positions'][1]:
                    mask_ab[m]=1
                    
        dpr_df.at[i,'mask_ab']=' '.join(str(e) for e in mask_ab)
        dpr_df.at[i,'mask_p']=' '.join(str(e) for e in mask_p)
        
    dpr_df_new = dpr_df.filter(['roberta_token','roberta_idx','label1',
                                    'mask_ab', 'mask_p','first_token','sequence','label2'], axis=1)
        
    return dpr_df_new


dpr_train_new=return_new_dpr_df(dpr_train)
dpr_test_new=return_new_dpr_df(dpr_test)

# write a pandas dataframe to csv file without row index
os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask\dpr')
dpr_train_new.to_csv("dpr_train_bert_2mask.csv", index=False,encoding='utf-8-sig')
dpr_test_new.to_csv("dpr_test_bert_2mask.csv", index=False,encoding='utf-8-sig')


os.chdir('E:/Google Drive/Continental/coref-multitask/dpr')
dpr_test_bert = pd.read_csv("dpr_test_bert_2mask.csv")
dpr_test_bert['label2'] = dpr_test_bert['label2'].astype(int)
dpr_test_bert.label2.unique()
dpr_test_bert.to_csv("dpr_test_bert_2mask_withindex.csv", index=True,encoding='utf-8-sig')
