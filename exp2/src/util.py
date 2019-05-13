import pandas as pd
import numpy as np
import random
import matplotlib
from matplotlib import pyplot as plt
import operator

#本文件主要用来做feature分析

#train set labels 'reviewerID', 'asin', 'reviewText', 'overall', 'votes_up', 'votes_all','label'
#test 'Id', 'reviewerID', 'asin', 'reviewText', 'overall'
global train_reviewerID,train_asin,train_reviewText,train_overall,train_votes_up,train_votes_all,train_label
global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall

def count_words(line):
    line = line.split(' ')
    cnt = 0
    for l in line:
        if l == '':
            continue
        if 'a' < l[0] < 'z' or 'A' < l[0] < 'Z':
            cnt += 1
    return cnt

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(list0,list1,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
    ax0.hist(list0,5,density=True)
    ax1.hist(list1,5,density=True)
    ax0.set_title('label-0 text length distribution')
    ax1.set_title('label-1 text length distribution')
    ax0.set_xlim(Xmin,Xmax)
    ax0.set_ylim(Ymin,Ymax)
 
    ax1.set_xlim(Xmin,Xmax)
    ax1.set_ylim(Ymin,Ymax)

    fig.subplots_adjust(hspace=0.4)
    plt.show()

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
    vote_up_one = 0.0
    vote_all_one = 0.0
    overall_one = 0.0
    text_len_one = 0.0
    word_one = 0.0
    vote_up_zero = 0.0
    vote_all_zero = 0.0
    overall_zero = 0.0
    text_len_zero = 0.0
    word_zero = 0.0
    id_len_zero = 0.0
    id_len_one = 0.0
     
    one_cnt = 0.0
    zero_cnt = 0.0

    one_len = []
    zero_len = []
    one_dic = {}
    zero_dic = {}
    asin_dic_one = {}
    asin_dic_zero = {}
    for i in range(len(train_label)):
        if train_label[i] == 1:
            one_len.append(train_votes_all[i])
            # if train_asin[i] in asin_dic_one.keys():
            #     asin_dic_one[train_asin[i]] += 1
            # else:
            #     asin_dic_one[train_asin[i]] = 1
        #     t = train_reviewText[i]
        #     t = t.replace('.',' ')
        #     t = t.replace('!',' ')
        #     t = t.replace(',',' ')
        #     t = t.replace('?',' ')
        #     t = t.replace('(',' ')
        #     t = t.replace(')',' ')
        #     t = t.replace('-',' ')
        #     t = t.replace('&',' ')
        #     t = t.replace(';',' ')
        #     t = t.lower()
        #     t = t.split(' ')
        #     t = list(set(t))
        #     for tt in t:
        #         if tt == '':
        #             continue
        #         else:
        #             word_one += 1.0 
        #             if tt in one_dic.keys():
        #                 one_dic[tt] += 1
        #             else:
        #                 one_dic[tt] = 1
        #     # vote_up_one += train_votes_up[i]
        #     # vote_all_one += train_votes_all[i]
            # overall_one += train_overall[i]
        #     # text_len_one += len(train_reviewText[i])
        #     # one_len.append(len(train_reviewText[i]))
        #     one_cnt += 1.0
            # word_one += count_words(train_reviewText[i])
            # id_len_one += len(train_asin[i])
        else:
            zero_len.append(train_votes_all[i])
        #     if train_asin[i] in asin_dic_zero.keys():
        #         asin_dic_zero[train_asin[i]] += 1
        #     else:
        #         asin_dic_zero[train_asin[i]] = 1
    # for i in asin_dic_zero.keys():
    #     asin_dic_zero[i] = float(asin_dic_zero[i])/float(train_label.count(0))
    # for i in asin_dic_one.keys():
    #     asin_dic_one[i] = float(asin_dic_one[i])/float(train_label.count(1))
    # delta_dic = {}
    # ave = 0.0
    # for i in asin_dic_zero.keys():
    #     if i in asin_dic_one.keys():
    #         delta = abs(asin_dic_zero[i] - asin_dic_one[i])
    #     else:
    #         delta = abs(asin_dic_zero[i])
    #     delta_dic[i] = delta
    #     ave += delta
    # for i in one_dic.keys():
    #     if i not in zero_dic.keys():
    #         delta = abs(one_dic[i])
    #         delta_dic[i] = delta
    #         ave+=delta
    # print(ave/float(len(set(train_asin))))
    # delta_dic = sorted(delta_dic.items(),key=operator.itemgetter(1),reverse=True)
            # t = train_reviewText[i]
            # t = t.replace('.',' ')
            # t = t.replace('!',' ')
            # t = t.replace(',',' ')
            # t = t.replace('?',' ')
            # t = t.replace('(',' ')              
            # t = t.replace(')',' ')
            # t = t.replace('-',' ')
            # t = t.replace('&',' ')
            # t = t.replace(';',' ')
            # t = t.lower()
            # t = t.split(' ')
            # t = list(set(t))
            # for tt in t:
            #     if tt == '':
            #         continue
            #     else:   
            #         word_zero += 1.0                  
            #         if tt in zero_dic.keys():
            #             zero_dic[tt] += 1
            #         else:
            #             zero_dic[tt] = 1
            # # vote_up_zero += train_votes_up[i]
            # # vote_all_zero += train_votes_all[i]
            # # overall_zero += train_overall[i]
            # # text_len_zero += len(train_reviewText[i])
            # # zero_len.append(len(train_reviewText[i]))
            # zero_cnt += 1.0
            # word_zero += count_words(train_reviewText[i])
            # id_len_zero += len(train_asin[i])
    # for i in zero_dic.keys():
    #     zero_dic[i] = float(zero_dic[i])/float(train_label.count(0))
    # for i in one_dic.keys():
    #     one_dic[i] = float(one_dic[i])/float(train_label.count(1))
    # ave = 0.0
    # cnt_dif = 0
    # delta_dic = {}
    # for i in zero_dic.keys():
    #     if i in one_dic.keys():
    #         delta = abs(zero_dic[i] - one_dic[i])
    #     else:
    #         delta = abs(zero_dic[i])
    #     cnt_dif += 1
    #     if delta > 1e-5:
    #         delta_dic[i] = delta
    #     ave += delta
    # for i in one_dic.keys():
    #     if i not in zero_dic.keys():
    #         delta = abs(one_dic[i])
    #         if delta > 1e-5:
    #             delta_dic[i] = delta
    #         cnt_dif += 1
    #         ave += delta
    # print(ave/cnt_dif)
    # delta_dic = sorted(delta_dic.items(),key=operator.itemgetter(1),reverse=True)
    y1 = []
    y0 = []
    x = []
    # delta_dic_t = delta_dic[:200]
    # for (k,v) in delta_dic_t:
    #     x.append(k)
    #     one = 0
    #     if k in asin_dic_one.keys():
    #         one = asin_dic_one[k]
    #     zero = 0
    #     if k in asin_dic_zero.keys():
    #         zero = asin_dic_zero[k]
    #     y0.append(zero)
    #     y1.append(one)
    x = [i for i in range(0,100)]
    for i in range(0,100):
        y0.append(zero_len.count(i)/float(len(zero_len)))
        y1.append(one_len.count(i)/float(len(one_len)))
    idx = np.arange(100)
    # print(x)
    width = 0.4
    print(y0)
    rects1 = plt.bar(idx, y0,width,color='red', label='one')
    rects2 = plt.bar(idx + width, y1,width,color='blue',label='zero')
    plt.xticks(idx + width, x)
    plt.title('overall distribution')
    plt.show()

    # zero_dic = sorted(zero_dic.items(),key=operator.itemgetter(1),reverse=True)
    # one_dic = sorted(one_dic.items(),key=operator.itemgetter(1),reverse=True)
    # print(delta_dic[:30])
    # print(zero_dic[:30])
    # print(one_dic[:30])

    # print(vote_up_one/one_cnt,vote_all_one/one_cnt,overall_one/one_cnt,text_len_one/one_cnt,word_one/one_cnt,id_len_one/one_cnt)
    # print(vote_up_zero/zero_cnt,vote_all_zero/zero_cnt,overall_zero/zero_cnt,text_len_zero/zero_cnt,word_zero/zero_cnt,id_len_zero/zero_cnt)
    # draw_hist(zero_len,one_len,'overall distibution','overall','frequency',Xmin=0,Xmax=5,Ymin=0,Ymax=1)

def read_test_data():
    global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall
    test_df = pd.read_csv('../data/test.csv', sep='\t')
    test_overall = list(map(float,test_df.loc[:,'overall']))
    test_Id = list(map(str,test_df.loc[:,'Id']))
    test_reviewerID = list(map(str,test_df.loc[:,'reviewerID']))
    test_asin = list(map(str,test_df.loc[:,'asin']))
    test_reviewText = list(map(str,test_df.loc[:,'reviewText']))

read_train_data()
a = ['dvd', 'movie', 'as', 'are', 'on', 'from', 'by', 'with', 'great', 'also', 'an', 'for', 'one', 'has', 'in', 'who', 'well', 'set', 'his', 'very', 'best', 'first', 'which', 'their', 'her', 'is', 'years', 'at', 'all', 'quality', 'most', 'will', 'music', 'more', 'time', 'of', 'each', 'series', 'when', 'some', 'two', 'up', 'other', 'that', 'you', 'wonderful', 'excellent', 'video', 'these', 'both', 'love', 'be', 'she', 'show', 'have', 'can', 'many', 'new', 'to', 'still', 'bad', 'sound', 'there', 'own', 'episodes', 'life', 'young', 'performance', 'a', 'and', 'release', 'shows', 'while', 'highly', 'been', 'beautiful', 'released', 'them', 'but', 'collection', 'only', 'my', 'may', 'out', 'its', 'work', 'this', 'than', 'episode', 'few', "it's", 'tv', 'production', 'features', 'classic', 'always', 'favorite', 'old', 'family', 'he', 'disc', 'back', 'perfect', 'version', 'here', 'later', 'during', 'cast', 'season', 'full', 'three', 'performances', 'recommend', 'it', 'although', 'picture', 'home', 'early', 'including', 'must', 'get', 'into', 'boring', 'good', 'see', 'especially', 'quite', 'find', 'available', '"the', 'before', 'included', 'ever', 'not', 'over', 'between', 'quot', 'little', 'seen', 'plot', 'dvds', 'so', 'those', 'sets', 'audio', 'does', 'they', 'several', 'different', 'long', 'now', 'played', 'live', 'such', 'through', 'after', 'fans', 'worst', 'times', 'makes',\
    'about', 'finally', 'workout', 'role', 'recommended', 'together', 'original', 'song', 'day', 'extras', 'superb', 'world', 'worth', 'though', 'often', 'stupid', 'heart', 'true', 'high', 'price', 'part', 'enjoy', 'movies', 'films', 'along', 'television', 'however', '"', 'four', 'plays', "didn't", 'late', 'second', 'waste', 'fun', 'songs', 'others', 'without', 'or', '1']