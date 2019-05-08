import pandas as pd
import numpy as np
import random

#train set labels 'reviewerID', 'asin', 'reviewText', 'overall', 'votes_up', 'votes_all','label'
#test 'Id', 'reviewerID', 'asin', 'reviewText', 'overall'
global train_reviewerID,train_asin,train_reviewText,train_overall,train_votes_up,train_votes_all,train_label
global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall

def read_train_data():
    global train_reviewerID,train_asin,train_reviewText,train_overall,train_votes_up,train_votes_all,train_label
    train_df = pd.read_csv('../data/train.csv', sep='\t')
    train_reviewerID = list(map(str,train_df.loc[:,'reviewerID']))
    train_asin = list(map(str,train_df.loc[:,'asin']))
    train_reviewText = list(map(str,train_df.loc[:,'reviewText']))
    train_overall = list(map(float,train_df.loc[:,'overall']))
    train_votes_up = list(map(int,train_df.loc[:,'votes_up']))
    train_votes_all = list(map(int,train_df.loc[:,'votes_all']))
    train_label = list(map(int,train_df.loc[:,'label']))

def read_test_data():
    global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall
    test_df = pd.read_csv('../data/test.csv', sep='\t')
    test_overall = list(map(float,test_df.loc[:,'overall']))
    test_Id = list(map(str,test_df.loc[:,'Id']))
    test_reviewerID = list(map(str,test_df.loc[:,'reviewerID']))
    test_asin = list(map(str,test_df.loc[:,'asin']))
    test_reviewText = list(map(str,test_df.loc[:,'reviewText']))

def write_result(item_id,predicted):
    result = pd.DataFrame({'Id':item_id,'predicted':predicted})
    result.to_csv('../output/result.csv',index = False,sep = ',')

def count_words(line):
    line = line.split(' ')
    return len(line)
