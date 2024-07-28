

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


os.chdir('D:/Google Drive/Continental/coref-multitask/gap')
gap_train = pd.read_csv("gap-development.tsv", sep='\t')
gap_dev = pd.read_csv("gap-validation.tsv", sep='\t')
gap_test = pd.read_csv("gap-test.tsv", sep='\t')


gap_test_female = pd.read_csv("gap-test_female_only.tsv", sep='\t')
gap_test_male = pd.read_csv("gap-test_male_only.tsv", sep='\t')

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel


os.chdir('D:/Google Drive/Continental/coref-multitask')
BERT_PATH = './bert-large-uncased' # the path of your downloaded pre-trained language model
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)


i=5
test_sen=gap_train.at[i,'Text']
roberta_idx=tokenizer.encode(test_sen)
tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))


test_sen=gap_train.at[i,'Text']
gap_train.at[i,'sequence']=test_sen

roberta_idx=tokenizer.encode(test_sen)
# if the list is of integers, convert the elements before joining them
gap_train.at[i,'roberta_idx']=' '.join(str(e) for e in roberta_idx)

tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
gap_train.at[i,'roberta_token']=' '.join(tokenized_testsen)


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


gap_train.at[i,'original_position_list']=original_position_list
gap_train.at[i,'first_token']=' '.join(str(e) for e in first_token_list)


# if want to insert list into cells, must set column type as object
gap_train['original_position_list']=""
gap_train['original_position_list']=gap_train['original_position_list'].astype('object')
gap_train['A_words_positions']=""
gap_train['A_words_positions']=gap_train['A_words_positions'].astype('object')
gap_train['B_words_positions']=""
gap_train['B_words_positions']=gap_train['B_words_positions'].astype('object')
gap_train['label1']=""
gap_train['label1']=gap_train['label1'].astype('object')
for i in range(gap_train.shape[0]):
    test_sen=gap_train.at[i,'Text']
    gap_train.at[i,'sequence']=test_sen
    
    roberta_idx=tokenizer.encode(test_sen)
    # if the list is of integers, convert the elements before joining them
    gap_train.at[i,'roberta_idx']=' '.join(str(e) for e in roberta_idx)
    
    tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
    gap_train.at[i,'roberta_token']=' '.join(tokenized_testsen)
    
    
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

    gap_train.at[i,'original_position_list']=original_position_list
    gap_train.at[i,'first_token']=' '.join(str(e) for e in first_token_list)
    
    
    pronoun_offset=gap_train.at[i,'Pronoun-offset']
    gap_train.at[i,'pronoun_word_position']=test_sen[:pronoun_offset].count(' ')
    
    A_offset=gap_train.at[i,'A-offset']
    A_start_word_position=test_sen[:A_offset].count(' ')
    A_num_of_spaces=gap_train.at[i,'A'].count(' ')
    A_end_word_position=A_start_word_position+A_num_of_spaces
    A_words_positions=[A_start_word_position,A_end_word_position]
    gap_train.at[i,'A_words_positions']=A_words_positions
    
    B_offset=gap_train.at[i,'B-offset']
    B_start_word_position=test_sen[:B_offset].count(' ')
    B_num_of_spaces=gap_train.at[i,'B'].count(' ')
    B_end_word_position=B_start_word_position+B_num_of_spaces
    B_words_positions=[B_start_word_position,B_end_word_position]
    gap_train.at[i,'B_words_positions']=B_words_positions
    
    label1_list=[0]*len(tokenized_testsen)
    for m in range(len(tokenized_testsen)):
        if gap_train.at[i,'original_position_list'][m]==gap_train.at[i,'pronoun_word_position']:
            label1_list[m]=1
        elif gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'A_words_positions'][0] \
        and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'A_words_positions'][1]:
            label1_list[m]=1
        elif gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'B_words_positions'][0] \
        and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'B_words_positions'][1]:
            label1_list[m]=1

    gap_train.at[i,'label1']=' '.join(str(e) for e in label1_list)
     
# repeat each row for three times
gap_train=gap_train.loc[gap_train.index.repeat(3)].reset_index(drop=True)

for i in range(gap_train.shape[0]):
    # A-P case
    if i%3==0:
        gap_train.at[i,'label2']=int(gap_train.at[i,'A-coref'])
    # B-P case
    if i%3==1:
        gap_train.at[i,'label2']=int(gap_train.at[i,'B-coref'])
    # A-B case
    if i%3==2:
        gap_train.at[i,'label2']=-1
        
        
    test_sen=gap_train.at[i,'Text']
    tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
    mask_list=[0]*len(tokenized_testsen)    
    # A-P case
    if i%3==0:
        for m in range(len(tokenized_testsen)):
            if gap_train.at[i,'original_position_list'][m]==gap_train.at[i,'pronoun_word_position']:
                mask_list[m]=1
            elif gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'A_words_positions'][0] \
            and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'A_words_positions'][1]:
                mask_list[m]=1
    # B-P case
    elif i%3==1:
        for m in range(len(tokenized_testsen)):
            if gap_train.at[i,'original_position_list'][m]==gap_train.at[i,'pronoun_word_position']:
                mask_list[m]=1
            elif gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'B_words_positions'][0] \
            and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'B_words_positions'][1]:
                mask_list[m]=1
    # A-B case
    elif i%3==2:
        for m in range(len(tokenized_testsen)):
            if gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'A_words_positions'][0] \
            and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'A_words_positions'][1]:
                mask_list[m]=1
            elif gap_train.at[i,'original_position_list'][m]>=gap_train.at[i,'B_words_positions'][0] \
            and gap_train.at[i,'original_position_list'][m]<=gap_train.at[i,'B_words_positions'][1]:
                mask_list[m]=1
                
    gap_train.at[i,'mask']=' '.join(str(e) for e in mask_list)

#gap_train.to_csv("gap_train.csv", index=False)

gap_train_new = gap_train.filter(['roberta_token','roberta_idx','label1',
                                'mask','first_token','sequence','label2'], axis=1)

# write a pandas dataframe to csv file without row index
gap_train_new.to_csv("gap_train_new_bert.csv", index=False)       


def return_new_gap_df(gap_df_original):
    # if want to insert list into cells, must set column type as object
    gap_df=gap_df_original.copy(deep=True)
    gap_df['original_position_list']=""
    gap_df['original_position_list']=gap_df['original_position_list'].astype('object')
    gap_df['A_words_positions']=""
    gap_df['A_words_positions']=gap_df['A_words_positions'].astype('object')
    gap_df['B_words_positions']=""
    gap_df['B_words_positions']=gap_df['B_words_positions'].astype('object')
    gap_df['label1']=""
    gap_df['label1']=gap_df['label1'].astype('object')
    
    for i in range(gap_df.shape[0]):
        test_sen=gap_df.at[i,'Text']
        gap_df.at[i,'sequence']=test_sen
        
        roberta_idx=tokenizer.encode(test_sen)
        # if the list is of integers, convert the elements before joining them
        gap_df.at[i,'roberta_idx']=' '.join(str(e) for e in roberta_idx)
        
        tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
        gap_df.at[i,'roberta_token']=' '.join(tokenized_testsen)
        
        
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

        gap_df.at[i,'original_position_list']=original_position_list
        gap_df.at[i,'first_token']=' '.join(str(e) for e in first_token_list)
        
        
        pronoun_offset=gap_df.at[i,'Pronoun-offset']
        gap_df.at[i,'pronoun_word_position']=test_sen[:pronoun_offset].count(' ')
        
        A_offset=gap_df.at[i,'A-offset']
        A_start_word_position=test_sen[:A_offset].count(' ')
        A_num_of_spaces=gap_df.at[i,'A'].count(' ')
        A_end_word_position=A_start_word_position+A_num_of_spaces
        A_words_positions=[A_start_word_position,A_end_word_position]
        gap_df.at[i,'A_words_positions']=A_words_positions
        
        B_offset=gap_df.at[i,'B-offset']
        B_start_word_position=test_sen[:B_offset].count(' ')
        B_num_of_spaces=gap_df.at[i,'B'].count(' ')
        B_end_word_position=B_start_word_position+B_num_of_spaces
        B_words_positions=[B_start_word_position,B_end_word_position]
        gap_df.at[i,'B_words_positions']=B_words_positions
        
        label1_list=[0]*len(tokenized_testsen)
        if gap_df.at[i,'A-coref']:
            for m in range(len(tokenized_testsen)):
                if gap_df.at[i,'original_position_list'][m]==gap_df.at[i,'pronoun_word_position']:
                    label1_list[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'A_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'A_words_positions'][1]:
                    label1_list[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'B_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'B_words_positions'][1]:
                    label1_list[m]=1
        elif gap_df.at[i,'B-coref']:
            for m in range(len(tokenized_testsen)):
                if gap_df.at[i,'original_position_list'][m]==gap_df.at[i,'pronoun_word_position']:
                    label1_list[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'A_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'A_words_positions'][1]:
                    label1_list[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'B_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'B_words_positions'][1]:
                    label1_list[m]=1

        gap_df.at[i,'label1']=' '.join(str(e) for e in label1_list)
         
         
    # repeat each row for three times
    gap_df=gap_df.loc[gap_df.index.repeat(2)].reset_index(drop=True)
    
    #gap_df = pd.DataFrame(np.repeat(gap_df.values, 3, axis=0), columns=gap_df.columns)
    print(gap_df.shape[0])
    
    

    for i in range(gap_df.shape[0]):
        
        # A-P case
        if i%2==0:
            gap_df.at[i,'label2']=int(gap_df.at[i,'A-coref'])
        # B-P case
        if i%2==1:
            gap_df.at[i,'label2']=int(gap_df.at[i,'B-coref'])
# =============================================================================
#         # A-B case
#         if i%3==2:
#             gap_df.at[i,'label2']=-1
# =============================================================================
        
        
            
        test_sen=gap_df.at[i,'Text']
        tokenized_testsen=tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sen))
        mask_ab=[0]*len(tokenized_testsen)
        mask_p=[0]*len(tokenized_testsen)
        # A-P case
        if i%2==0:
            for m in range(len(tokenized_testsen)):
                if gap_df.at[i,'original_position_list'][m]==gap_df.at[i,'pronoun_word_position']:
                    mask_p[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'A_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'A_words_positions'][1]:
                    mask_ab[m]=1
        # B-P case
        elif i%2==1:
            for m in range(len(tokenized_testsen)):
                if gap_df.at[i,'original_position_list'][m]==gap_df.at[i,'pronoun_word_position']:
                    mask_p[m]=1
                elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'B_words_positions'][0] \
                and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'B_words_positions'][1]:
                    mask_ab[m]=1
# =============================================================================
#         # A-B case
#         elif i%3==2:
#             for m in range(len(tokenized_testsen)):
#                 if gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'A_words_positions'][0] \
#                 and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'A_words_positions'][1]:
#                     mask_list[m]=1
#                 elif gap_df.at[i,'original_position_list'][m]>=gap_df.at[i,'B_words_positions'][0] \
#                 and gap_df.at[i,'original_position_list'][m]<=gap_df.at[i,'B_words_positions'][1]:
#                     mask_list[m]=1
# =============================================================================
                    
        gap_df.at[i,'mask_ab']=' '.join(str(e) for e in mask_ab)
        gap_df.at[i,'mask_p']=' '.join(str(e) for e in mask_p)
        
    gap_df_new = gap_df.filter(['roberta_token','roberta_idx','label1',
                                    'mask_ab', 'mask_p','first_token','sequence','label2'], axis=1)
        
    return gap_df_new


gap_train_new=return_new_gap_df(gap_train)
gap_dev_new=return_new_gap_df(gap_dev)
gap_test_new=return_new_gap_df(gap_test)

gap_test_female_new=return_new_gap_df(gap_test_female)
gap_test_male_new=return_new_gap_df(gap_test_male)

# =============================================================================
# # write a pandas dataframe to csv file without row index
# os.chdir('E:/Google Drive/Continental/coref-multitask/gap-data')
# gap_train_new.to_csv("gap_train_new_bert.csv", index=False,encoding='utf-8-sig')
# gap_dev_new.to_csv("gap_dev_new_bert.csv", index=False,encoding='utf-8-sig')
# gap_test_new.to_csv("gap_test_new_bert.csv", index=False,encoding='utf-8-sig')
# =============================================================================


# write a pandas dataframe to csv file without row index
os.chdir('D:/Google Drive/Continental/coref-multitask/gap')
gap_train_new.to_csv("gap_train_bert_2mask_old.csv", index=False,encoding='utf-8-sig')

# write a pandas dataframe to csv file without row index
os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask\gap')
gap_train_new.to_csv("gap_train_bert_2mask.csv", index=False,encoding='utf-8-sig')
gap_dev_new.to_csv("gap_dev_bert_2mask.csv", index=False,encoding='utf-8-sig')
gap_test_new.to_csv("gap_test_bert_2mask.csv", index=False,encoding='utf-8-sig')

gap_test_female_new.to_csv("gap_test_female_bert_2mask.csv", index=False,encoding='utf-8-sig')
gap_test_male_new.to_csv("gap_test_male_bert_2mask.csv", index=False,encoding='utf-8-sig')

os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask\gap')
gap_test_bert = pd.read_csv("gap_train_bert_2mask.csv")
gap_test_bert['label2'] = gap_test_bert['label2'].astype(int)
gap_test_bert.label2.unique()
gap_test_bert.to_csv("gap_train_bert_2mask.csv", index=False,encoding='utf-8-sig')

gap_test_bert = pd.read_csv("gap_test_bert_2mask.csv")
gap_test_bert['label2'] = gap_test_bert['label2'].astype(int)
gap_test_bert.label2.unique()
gap_test_bert.to_csv("gap_test_bert_2mask.csv", index=False,encoding='utf-8-sig')

gap_test_bert = pd.read_csv("gap_test_male_bert_2mask.csv")
gap_test_bert['label2'] = gap_test_bert['label2'].astype(int)
gap_test_bert.label2.unique()
gap_test_bert.to_csv("gap_test_male_bert_2mask.csv", index=False,encoding='utf-8-sig')

gap_test_bert = pd.read_csv("gap_test_female_bert_2mask.csv")
gap_test_bert['label2'] = gap_test_bert['label2'].astype(int)
gap_test_bert.label2.unique()
gap_test_bert.to_csv("gap_test_female_bert_2mask.csv", index=False,encoding='utf-8-sig')

os.chdir(r'D:\Google Drive Streaming\My Drive\Continental\coref-multitask\gap')
gap_test_bert = pd.read_csv("gap_test_bert_2mask.csv")
gap_test_bert['label2'] = gap_test_bert['label2'].astype(int)
gap_test_bert.label2.unique()
gap_test_bert.to_csv("gap_test_bert_2mask_withindex.csv", index=True,encoding='utf-8-sig')

# =============================================================================
# # write train set with bio to tsv file without index
# df_conll2012train.to_csv("conll_train_withbio.tsv", sep="\t", index=False)
# 
# # write dev set with bio to tsv file without index
# df_conll2012dev.to_csv("conll_dev_withbio.tsv", sep="\t", index=False)
# 
# # write test set with bio to tsv file without index
# df_conll2012test.to_csv("conll_test_withbio.tsv", sep="\t", index=False)
# =============================================================================



# =============================================================================
# # write a pandas dataframe to csv file without row index
# df_conll2012train.to_csv("conll_train_new.csv", index=False)
# =============================================================================


# listoflists=df_conll2012train.at[2,'sentences']

# ' '.join(' '.join(sub) for sub in listoflists)





# stanford_tagger(df_conll2012train.at[2,'sentences'])
