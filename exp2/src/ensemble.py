from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import random

global train_reviewerID,train_asin,train_reviewText,train_overall,train_votes_up,train_votes_all,train_label
global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall
DECISION_TREE = 1
SVM = 2

TEST_ON_TRAIN_SET = False
CLASSFIER_TO_USE = DECISION_TREE

TRAIN_SET_SIZE = 57039
if(TEST_ON_TRAIN_SET):
    TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 11208


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
    print(train_label.count(1),train_label.count(0))

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

def get_test_set():
    global test_Id,test_reviewerID,test_asin,test_reviewText,test_overall
    test_set = []
    for i in range(0,len(test_reviewerID)):
        one_set = []
        # one_set.append(test_reviewerID[i])
        # one_set.append(test_asin[i])
        one_set.append(test_overall[i])
        # one_set.append(train_set['votes_up'][i])
        # one_set.append(train_set['votes_all'][i])
        one_set.append(count_words(test_reviewText[i]))
        test_set.append(one_set)
    return test_set

#从指定范围的训练集中取出随机大小的训练集
def random_pick_train_set(size,start,end):
    global train_reviewerID,train_asin,train_reviewText,train_overall,train_votes_up,train_votes_all,train_label
    #从[start,end)中随机取size个随机自然数
    big_set = np.arange(start,end)
    random.shuffle(big_set)
    temp = big_set[0:size]
    #取出对应训练集
    small_train_set = []
    small_label_set = []
    weights = []
    cnt_1 = float(train_label.count(1))/float(len(train_label))
    cnt_0 = float(train_label.count(0))/float(len(train_label))
    for i in temp:
        one_set = []
        # one_set.append(train_reviewerID[i])
        # one_set.append(train_asin[i])
        one_set.append(train_overall[i])
        # one_set.append(train_set['votes_up'][i])
        # one_set.append(train_set['votes_all'][i])
        one_set.append(count_words(train_reviewText[i]))
        x =cnt_0
        if(train_label[i] == 1):
            x = cnt_1
        weights.append(float(train_votes_up[i])/float(train_votes_all[i]))
        small_train_set.append(one_set)
        small_label_set.append(train_label[i])
    # print(small_train_set)
    return small_train_set,small_label_set,temp,weights

def predict(classfier,my_train_set,my_label_set,my_test_set,weights):
    if classfier == DECISION_TREE:
        clf = DecisionTreeClassifier()
        print('train with decision tree')
    elif classfier == SVM:
        clf = SVC(gamma='auto')
        print('train with SVM')
    else:
        print('ERROR: NO SUCH CLASSFIER')
    clf.fit(my_train_set,my_label_set,weights)
    print('predicting...')
    res = clf.predict(my_test_set)
    print(list(res).count(1),list(res).count(0))
    return res

def check_result(predict_result,label_set):
    total_num = len(label_set)
    correct_cnt = 0
    print('predict_len',len(predict_result),len(label_set))
    for i in range(0,len(predict_result)):
        ans = label_set[i]
        # if(predict_result[i] == 1):
        #     print('cool',ans == predict_result[i],ans)
        if(abs(ans - float(predict_result[i])) < 0.5):
            correct_cnt += 1
    print('total num:',total_num,'correct_cnt',correct_cnt)
    return float(correct_cnt)/float(total_num)

def calculate_result(results,weights,num):
    result = []
    cnt_1 = 0
    cnt_0 = 0
    for i in range(0,len(results[0])):
        temp = 0.0
        for j in range(0,num):
            temp += float(results[j][i])/float(num)
        if(temp > 0.99999):
            temp = 0.99999
        if(temp > 0.5):
            cnt_1+=1
        else:
            cnt_0 += 1

        result.append(temp)
    print('count 0 and count 1',cnt_0,cnt_1)
    return result

if __name__ == '__main__':
    read_test_data()
    read_train_data()
    print('data loaded')
    results = []
    pick_test_set = get_test_set()
    if(TEST_ON_TRAIN_SET):
        print('TEST ON TRAIN SET')
        pick_test_set,pick_test_label_set,testid_set,t_weights = random_pick_train_set(5000,TRAIN_SET_SIZE,57039)
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    pick_train_set,pick_label_set,id_set,weights = random_pick_train_set(20000,0,TRAIN_SET_SIZE)
    results.append(predict(CLASSFIER_TO_USE,pick_train_set,pick_label_set,pick_test_set,weights))
    result = calculate_result(results,[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],11)
    if(not TEST_ON_TRAIN_SET):
        write_result(test_Id,result)
    else:
        print(check_result(result,pick_test_label_set))
    print(len(result))
    # print(check_result(result,pick_test_label_set))