from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.deprecated.word2vec import Word2Vec
import pandas as pd
import numpy as np
import random
import math
import pickle
import sys
import word2vec

global train_df,train_label,train_data,model
global test_Id,test_data,test_df
model = None
test_Id = None
test_data = None
test_df = None
train_df = None
train_label = None
train_data = None

DECISION_TREE = 1
SVM = 2
BAGGING = 3
ADABOOST_M1 = 4
EQUAL_WEIGHT = 5

BAGGING_T = 21
ADABOOST_M1_T = 11
TEST_ON_TRAIN_SET = False
CLASSFIER_TO_USE = SVM
ENSEMBLE_WAY = BAGGING
SAMPLE_WEIGHT = EQUAL_WEIGHT
USE_TEXT_VECTOR = True


TRAIN_SET_SIZE = 57039
if(TEST_ON_TRAIN_SET):
    TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 11208

def openPickleFile(path_):
    invertedDoc = {}
    try:
        pkl_file = open(path_, 'rb')
        invertedDoc = pickle.load(pkl_file)
        pkl_file.close()
        print(path_ + ' read successfully!')
    except:
        print(path_ + ' read failed!')
        print("Unexpected error:", sys.exc_info()[0])

    return invertedDoc

def savePickleFile(data, path_) :
    try :
        output = open(path_, 'wb')
        pickle.dump(data, output)
        output.close()
        print(path_ + ' save successfully!')
    except :
        print(path_ + ' save failed!')
        print("Unexpected error:", sys.exc_info()[0])

def read_train_and_test_data():
    global train_label,train_data,train_df
    train_df = pd.read_csv('../data/train.csv', sep='\t')
    train_label = list(map(int,train_df.loc[:,'label']))

    global test_Id,test_data,test_df
    test_df = pd.read_csv('../data/test.csv', sep='\t')
    test_Id = list(map(str,test_df.loc[:,'Id']))
    print('data loaded')

    test_data = []
    train_data = []
    train_vec = []
    test_vec = []
    if(USE_TEXT_VECTOR):
        #countervec
        global model
        model = TfidfVectorizer(analyzer='word', stop_words='english')
        train_vec = model.fit_transform(train_df['reviewText'])
        test_vec = model.transform(test_df['reviewText'])
        train_data = hstack([train_vec, np.mat(train_df['overall']).reshape((len(train_df), 1))], format='csr')
        test_data = hstack([test_vec, np.mat(test_df['overall']).reshape((len(test_df), 1))],format='csr')
        # print(train_data)
        print('text vec built finished')
        #word2vec
        # model = Word2Vec(train_df['reviewerText']+test_df['reviewText'],size=50, window=5, min_count=1, workers=4)
        # for i in train_df['reviewerText']:
        #     t = get_words(i)
        #     #求每个词的词向量然后求均值
        #     v = None
        #     cnt = 0
        #     for tt in t:
        #         if v == None:
        #             v = model.wv[tt]
        #             cnt = 1
        #         else:
        #             v += model.wv[tt]
        #             cnt += 1
        #     print(cnt)
        #     if not cnt == 0:
        #         v /= cnt
        #     else:
        #         v = [0 for i in range(50)]
        #     train_data.append(v)
        # for i in test_df['reviewText']:
        #     t = get_words(i)
        #     #求每个词的词向量然后求均值
        #     v = None
        #     cnt = 0
        #     for tt in t:
        #         if v == None:
        #             v = model.wv[tt]
        #             cnt = 1
        #         else:
        #             v += model.wv[tt]
        #             cnt += 1
        #     print(cnt)
        #     if not cnt == 0:
        #         v /= cnt
        #     else:
        #         v = [0 for i in range(50)]
        #     test_data.append(v)
    #构造训练数据集和测试数据集
    for i in range(len(train_label)):
        one_set = []
        if(USE_TEXT_VECTOR):
            None
        else:
            if(CLASSFIER_TO_USE != SVM):
                one_set.append(train_df['overall'][i])
                one_set.append(train_df['asin'][i])
                one_set.append(train_df['reviewerID'][i])
                one_set.append(len(train_df['reviewerText'][i]))
                train_data.append(one_set)
            else:
                one_set.append(len(train_df['reviewerText'][i]))
                one_set.append(train_df['overall'][i])
                train_data.append(one_set)
    for i in range(len(test_Id)):
        one_set = []
        if(USE_TEXT_VECTOR):
            None
        else:
            if(CLASSFIER_TO_USE != SVM):
                one_set.append(test_df['overall'][i])
                one_set.append(test_df['asin'][i])
                one_set.append(test_df['reviewerID'][i])
                one_set.append(len(test_df['reviewText'][i]))
                test_data.append(one_set)
            else:
                one_set.append(len(test_df['reviewText'][i]))
                one_set.append(test_df['overall'][i])
                test_data.append(one_set)
    
    print(train_label.count(1),train_label.count(0))


def write_result(item_id,predicted):
    result = pd.DataFrame({'Id':item_id,'predicted':predicted})
    result.to_csv('../output/result.csv',index = False,sep = ',')
    print('result written')

def count_words(line):
    line = line.split(' ')
    return len(line)

def get_words(t):
    t = t.replace('.',' ')
    t = t.replace('!',' ')
    t = t.replace(',',' ')
    t = t.replace('?',' ')
    t = t.replace('(',' ')              
    t = t.replace(')',' ')
    t = t.replace('-',' ')
    t = t.replace('&',' ')
    t = t.replace(';',' ')
    t = t.lower()
    t = t.split(' ')
    t = list(set(t))
    return t

def get_test_set():
    global test_data
    return test_data

#从指定范围的训练集中取出随机大小的训练集
def get_train_set(size,start,end):
    global train_df,train_label
    big_set = np.arange(start,end)
    temp = big_set[0:size]
    #取出对应训练集
    small_train_set = []
    small_label_set = []
    weights = []

    for i in temp:
        if(SAMPLE_WEIGHT == EQUAL_WEIGHT):
            # if(small_label_set)
            weights.append(float(1)/float(size))
        else:
            weights.append(float(train_df['votes_up'][i])/float(train_df['votes_all'][i]))
        # small_train_set.append(train_data[i].toarray())
        small_train_set.append(train_data[i])
        small_label_set.append(train_label[i])
    # small_train_set = csr_matrix(small_train_set)
    # print(small_train_set)
    normalize(weights)

    return small_train_set,small_label_set,weights

def random_pick_one_in_train_set(start=0,end=TRAIN_SET_SIZE):
    global train_data,train_label
    i = random.randint(start,end - 1)
    return i

def train_and_predict(classfier,my_train_set,my_label_set,my_test_set,weights = None):
    if classfier == DECISION_TREE:
        clf = DecisionTreeClassifier(min_samples_split=30,class_weight='balanced')
        print('train with decision tree')
        clf.fit(my_train_set,my_label_set,sample_weight=weights)
        print('predicting...')
        res = clf.predict(my_test_set)
    elif classfier == SVM:
        clf = LinearSVC()
        clf = CalibratedClassifierCV(clf, method='sigmoid', cv = 3)
        print('train with SVM')
        clf.fit(my_train_set,my_label_set)
        print('predicting...')
        res_ = clf.predict_proba(my_test_set)
        res = []
        for i in range(len(res_)):
            res.append(round(res_[i][1],0))
        # res = clf.predict(my_test_set)
    else:
        print('ERROR: NO SUCH CLASSFIER')
    print(list(res).count(1),list(res).count(0))
    # print(list(res).count(1),list(res).count(0))
    return res,clf

def predict_(classfier,clf,test_set):
    if classfier == DECISION_TREE:
        res = clf.predict(test_set)
        print('train with decision tree')
    elif classfier == SVM:
        # res = clf.predict(test_set)
        res_ = clf.predict_proba(test_set)
        res = []
        for i in range(len(res_)):
            res.append(round(res_[i][1],0))
    else:
        print('ERROR: NO SUCH CLASSFIER')
    print('predicting...')
    print(list(res).count(1),list(res).count(0))
    return res

def check_result(predict_result,label_set):
    total_num = len(label_set)
    correct_cnt = 0
    correct_one = 0
    correct_zero = 0

    for i in range(0,len(predict_result)):
        ans = label_set[i]
        # if(predict_result[i] == 1):
        #     print('cool',ans == predict_result[i],ans)
        if(abs(ans - float(predict_result[i])) < 0.5):
            correct_cnt += 1
            if(ans < 0.1):
                correct_zero += 1
            else:
                correct_one += 1
    print('correct_rate=',float(correct_cnt)/float(total_num),',correct_one_rate=',\
        float(correct_one)/float(label_set.count(1)),',correct_zero_rate=',float(correct_zero)/float(label_set.count(0)))
    return float(correct_cnt)/float(total_num)

def calculate_result(results,num,weights = None):
    result = []
    cnt_1 = 0
    cnt_0 = 0
    weights_sum = 0.0

    if weights != None:
        if len(weights) >= 2:
            for j in range(num):
                    weights_sum += weights[j]
            print(weights_sum)
        else:
            weights = None
    for i in range(0,len(results[0])):
        temp = 0.0
        if weights == None:
            for j in range(num):
                temp += float(results[j][i])/float(num)
                print(results[j][i],temp)
        else:
            for j in range(num):
                temp += float(results[j][i])*weights[j]
        if weights_sum > 0.001:
            temp /= weights_sum
        if(temp > 0.5):
            # a = random.randint(6,11)
            # temp = float(a)/float(11)
            cnt_1+=1
        else:
            # a = random.randint(0,5)
            # temp = float(a)/float(11)
            cnt_0 += 1
        if(temp > 0.99999):
            temp = 0.99999
        result.append(temp)
    print('count 0 and count 1',cnt_0,cnt_1)
    return result

def get_bagging_data():
    train_set = []
    my_label_set = []
    weight_set = []
    t_set = []
    for i in range(TRAIN_SET_SIZE):
        t_set.append(random_pick_one_in_train_set())
    if(USE_TEXT_VECTOR):
        global model
        revtext_t = []
        overall_t = []
        for i in t_set:
            revtext_t.append(train_df['reviewText'][i])
            overall_t.append(train_df['overall'][i])
            my_label_set.append(train_label[i])
        # model = TfidfVectorizer(analyzer='word', stop_words='english')
        vec_t = model.transform(revtext_t)
        train_set = hstack([vec_t, np.mat(overall_t).reshape((len(train_df), 1))], format='csr')
    else:
        for t in t_set:
            train_set.append(train_data[t])
            my_label_set.append(train_label[t])
    return train_set,my_label_set

def bagging(test_set,real_label_set = None):
    #build train_set
    print('bagging test begins')
    result_sets = []
    for i in range(0,BAGGING_T):
        print('******************bagging test ' + str(i) + '*********')
        train_set = label_set = None
        train_set,label_set = get_bagging_data()
        result_set,clf = train_and_predict(CLASSFIER_TO_USE,train_set,label_set,test_set)
        if(TEST_ON_TRAIN_SET):
            check_result(result_set,real_label_set)
        result_sets.append(result_set)
    result = calculate_result(result_sets,BAGGING_T)
    # print(result)
    return result

def normalize(list_):
    sum = 0.0
    for i in list_:
        sum += i
    for i in range(len(list_)):
        list_[i] /= sum
    return list_

def adaboost_m1(test_set,real_label_set = None):
    print('AdaBoost.M1 test begins')
    result_sets = []
    clf_weight_set = []
    final_t = 0
    # train_set,label_set,sample_weight_set = get_train_set(TRAIN_SET_SIZE,0,TRAIN_SET_SIZE)
    global train_data,train_label,test_data
    train_set = train_data
    label_set = train_label
    sample_weight_set = [ 1.0/len(label_set) for i in range(len(label_set))]
    one_set = []
    zero_set = []
    for i in range(len(label_set)):
            if label_set[i] == 1:
                one_set.append(i)
            else:
                zero_set.append(i)
    for i in range(0,ADABOOST_M1_T):
        print('************AdaBoost.M1 ' + str(i) + '***********')
        # print(len(train_set))
        result_set,clf = train_and_predict(CLASSFIER_TO_USE,train_set,label_set,train_set,sample_weight_set)
        check_result(result_set,label_set)
        e = 0.0
        for i in range(len(result_set)):
            # print(label_set[i],result_set[i])
            if(label_set[i] != result_set[i]):
                print(result_set[i],label_set[i])
                e += sample_weight_set[i]
        print('error rate',e)
        if e > 0.5:
            break
        final_t += 1
        beta = e/(1-e)

        #modify earch sample's weight
        for i in range(len(result_set)):
            if(label_set[i] == result_set[i]):
                # print(sample_weight_set[i],sample_weight_set[i]*beta)
                sample_weight_set[i] = sample_weight_set[i]*beta

        
        # one_sum = 0.0
        # zero_sum = 0.0
        # for i in one_set:
        #     one_sum += sample_weight_set[i]
        # for i in zero_set:
        #     zero_sum += sample_weight_set[i]
        # one_sum *= 2
        # zero_sum *= 2
        # for i in one_set:
        #     sample_weight_set[i] /= one_sum
        # for i in zero_set:
        #     sample_weight_set[i] /= zero_sum

        sample_weight_set = normalize(sample_weight_set)
        clf_weight_set.append(math.log(1/beta))
        res = predict_(CLASSFIER_TO_USE,clf,test_data)
        if(TEST_ON_TRAIN_SET):
            check_result(res,real_label_set)
        result_sets.append(res)

    result = calculate_result(result_sets,final_t,weights=clf_weight_set)
    return result

if __name__ == '__main__':
    read_train_and_test_data()
    results = []
    pick_test_set = get_test_set()
    pick_test_label_set = None
    # print(train_df['reviewerText'][0])
    if(TEST_ON_TRAIN_SET):
        print('TEST ON TRAIN SET')
        pick_test_set,pick_test_label_set,t_weights = get_train_set(7038,TRAIN_SET_SIZE,57039)
  
    result = []

    if ENSEMBLE_WAY == BAGGING:
        result = bagging(pick_test_set,pick_test_label_set)
    elif ENSEMBLE_WAY == ADABOOST_M1:
        result = adaboost_m1(pick_test_set,pick_test_label_set)
        # clf = BaggingClassifier(DecisionTreeClassifier(class_weight='balanced',min_samples_leaf=0.1))
        # train_set,label_set,sample_weight_set = get_train_set(TRAIN_SET_SIZE,0,TRAIN_SET_SIZE)
        # clf.fit(train_set,label_set)
        # result = clf.predict(pick_test_set)

    if(not TEST_ON_TRAIN_SET):
        write_result(test_Id,result)
    else:
        check_result(result,pick_test_label_set)
    # print(check_result(result,pick_test_label_set))