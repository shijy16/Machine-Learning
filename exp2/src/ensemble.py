from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.deprecated.word2vec import Word2Vec

import pandas as pd
import numpy as np
import random
import math
import pickle
import sys
import word2vec


global train_votes_up,train_votes_all,train_label,train_data
global test_Id,test_data
DECISION_TREE = 1
SVM = 2
BAGGING = 3
ADABOOST_M1 = 4
EQUAL_WEIGHT = 5

BAGGING_T = 21
ADABOOST_M1_T = 11
TEST_ON_TRAIN_SET = False
CLASSFIER_TO_USE = DECISION_TREE
ENSEMBLE_WAY = ADABOOST_M1
SAMPLE_WEIGHT = EQUAL_WEIGHT
USE_TEXT_VECTOR = True


TRAIN_SET_SIZE = 57039
if(TEST_ON_TRAIN_SET):
    TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 11208

keywords = ['dvd', 'movie', 'as', 'are', 'on', 'from', 'by', 'with', 'great', 'also', 'an', 'for', 'one', 'has', 'in', 'who', 'well', 'set', 'his', 'very', 'best', 'first', 'which', 'their', 'her', 'is', 'years', 'at', 'all', 'quality', 'most', 'will', 'music', 'more', 'time', 'of', 'each', 'series', 'when', 'some', 'two', 'up', 'other', 'that', 'you', 'wonderful', 'excellent', 'video', 'these', 'both', 'love', 'be', 'she', 'show', 'have', 'can', 'many', 'new', 'to', 'still', 'bad', 'sound', 'there', 'own', 'episodes', 'life', 'young', 'performance', 'a', 'and', 'release', 'shows', 'while', 'highly', 'been', 'beautiful', 'released', 'them', 'but', 'collection', 'only', 'my', 'may', 'out', 'its', 'work', 'this', 'than', 'episode', 'few', "it's", 'tv', 'production', 'features', 'classic', 'always', 'favorite', 'old', 'family', 'he', 'disc', 'back', 'perfect', 'version', 'here', 'later', 'during', 'cast', 'season', 'full', 'three', 'performances', 'recommend', 'it', 'although', 'picture', 'home', 'early', 'including', 'must', 'get', 'into', 'boring', 'good', 'see', 'especially', 'quite', 'find', 'available', '"the', 'before', 'included', 'ever', 'not', 'over', 'between', 'quot', 'little', 'seen', 'plot', 'dvds', 'so', 'those', 'sets', 'audio', 'does', 'they', 'several', 'different', 'long', 'now', 'played', 'live', 'such', 'through', 'after', 'fans', 'worst', 'times', 'makes', \
    'about', 'finally', 'workout', 'role', 'recommended', 'together', 'original', 'song', 'day', 'extras', 'superb', 'world', 'worth', 'though', 'often', 'stupid', 'heart', 'true', 'high', 'price', 'part', 'enjoy', 'movies', 'films', 'along', 'television', 'however', '"', 'four', 'plays', "didn't", 'late', 'second', 'waste', 'fun', 'songs', 'others', 'without', 'or', '1']

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
    global train_votes_up,train_votes_all,train_label,train_data
    train_df = pd.read_csv('../data/train.csv', sep='\t')
    train_reviewerID = list(map(str,train_df.loc[:,'reviewerID']))
    train_asin = list(map(str,train_df.loc[:,'asin']))
    train_reviewText = list(map(str,train_df.loc[:,'reviewText']))
    train_overall = list(map(float,train_df.loc[:,'overall']))
    train_votes_up = list(map(int,train_df.loc[:,'votes_up']))
    train_votes_all = list(map(int,train_df.loc[:,'votes_all']))
    train_label = list(map(int,train_df.loc[:,'label']))

    global test_Id,test_data
    test_df = pd.read_csv('../data/test.csv', sep='\t')
    test_overall = list(map(float,test_df.loc[:,'overall']))
    test_Id = list(map(str,test_df.loc[:,'Id']))
    test_reviewerID = list(map(str,test_df.loc[:,'reviewerID']))
    test_asin = list(map(str,test_df.loc[:,'asin']))
    test_reviewText = list(map(str,test_df.loc[:,'reviewText']))
    test_data = []
    train_data = []

    train_vec = []
    test_vec = []
    if(USE_TEXT_VECTOR):
        #tfdif
        # tfidf_model = TfidfVectorizer().fit(train_reviewText)
        # tfidf_model.fit(test_reviewText)
        # train_vec = tfidf_model.transform(train_reviewText)
        # train_data = csr_matrix(train_vec)
        # test_vec = tfidf_model.transform(test_reviewText)
        # test_data = csr_matrix(test_vec)
        
        #word2vec
        model = Word2Vec(train_reviewText+test_reviewText,size=50, window=5, min_count=1, workers=4)
        for i in train_reviewText:
            t = get_words(i)
            #求每个词的词向量然后求均值
            v = None
            cnt = 0
            for tt in t:
                if v == None:
                    v = model.wv[tt]
                    cnt = 1
                else:
                    v += model.wv[tt]
                    cnt += 1
            print(cnt)
            if not cnt == 0:
                v /= cnt
            else:
                v = [0 for i in range(50)]
            train_data.append(v)
        for i in test_reviewText:
            t = get_words(i)
            #求每个词的词向量然后求均值
            v = None
            cnt = 0
            for tt in t:
                if v == None:
                    v = model.wv[tt]
                    cnt = 1
                else:
                    v += model.wv[tt]
                    cnt += 1
            print(cnt)
            if not cnt == 0:
                v /= cnt
            else:
                v = [0 for i in range(50)]
            test_data.append(v)
    #构造训练数据集和测试数据集
    for i in range(len(train_label)):
        one_set = []
        if(USE_TEXT_VECTOR):
            # one_set.append(train_overall[i])
            # one_set.append(train_asin[i])
            # one_set.append(train_reviewerID[i])
            # one_set.append(train_data_t[i])
            # train_data.append(one_set)
            None
        
        else:
            if(CLASSFIER_TO_USE != SVM):
                one_set.append(train_overall[i])
                one_set.append(train_asin[i])
                one_set.append(train_reviewerID[i])
                one_set.append(len(train_reviewText[i]))
                train_data.append(one_set)
            else:
                one_set.append(len(train_reviewText[i]))
                one_set.append(train_overall[i])
                train_data.append(one_set)
    if(USE_TEXT_VECTOR):
        # train_data = csr_matrix(train_data)
        # test_vec = cv.transform(test_reviewText)
        test_reviewText = None
    for i in range(len(test_Id)):
        one_set = []
        if(USE_TEXT_VECTOR):
            # test_data.append(test_vec[i].toarray())
            # one_set.append(test_overall[i])
            # one_set.append(test_asin[i])
            # one_set.append(test_reviewerID[i])
            # one_set.append(test_data_t[i])
            # test_data.append(one_set)
            None
        else:
            if(CLASSFIER_TO_USE != SVM):
                one_set.append(test_overall[i])
                one_set.append(test_asin[i])
                one_set.append(test_reviewerID[i])
                one_set.append(len(test_reviewText[i]))
                test_data.append(one_set)
            else:
                one_set.append(len(test_reviewText[i]))
                one_set.append(test_overall[i])
                test_data.append(one_set)
    if(USE_TEXT_VECTOR):
        # test_data = csr_matrix(test_data)
        None
    
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
    global train_votes_up,train_votes_all,train_label,train_data
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
            weights.append(float(train_votes_up[i])/float(train_votes_all[i]))
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
    weight = float(train_votes_up[i])/float(train_votes_all[i])
    return train_data[i],train_label[i],weight

def train_and_predict(classfier,my_train_set,my_label_set,my_test_set,weights):
    if classfier == DECISION_TREE:
        clf = DecisionTreeClassifier(min_samples_split=30,class_weight='balanced')
        print('train with decision tree')
    elif classfier == SVM:
        clf = SVC(gamma='auto',kernel='linear',class_weight='balanced')
        print('train with SVM')
    else:
        print('ERROR: NO SUCH CLASSFIER')
   
    clf.fit(my_train_set,my_label_set,sample_weight=weights)
    print('predicting...')
    res = clf.predict(my_test_set)
    print(list(res).count(1),list(res).count(0))
    return res,clf

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
        for j in range(num):
                weights_sum += weights[j]
        print(weights_sum)
    for i in range(0,len(results[0])):
        temp = 0.0
        if weights == None:
            for j in range(num):
                temp += float(results[j][i])/float(num)
        else:
            for j in range(num):
                temp += float(results[j][i])*weights[j]
        if weights_sum > 0.001:
            temp /= weights_sum
        if(temp > 0.99999):
            temp = 0.99999
        if(temp > 0.5):
            cnt_1+=1
        else:
            cnt_0 += 1

        result.append(temp)
    print('count 0 and count 1',cnt_0,cnt_1)
    return result

def get_bagging_data():
    train_set = []
    my_label_set = []
    weight_set = []
    for i in range(TRAIN_SET_SIZE):
        train_t,label_t,weight_t = random_pick_one_in_train_set()
        train_set.append(train_t)
        my_label_set.append(label_t)
        weight_set.append(weight_t)
    return train_set,my_label_set,weight_set

def bagging(test_set,real_label_set = None):
    #build train_set
    print('bagging test begins')
    result_sets = []
    for i in range(0,BAGGING_T):
        print('******************bagging test ' + str(i) + '*********')
        train_set = label_set = weight_set = None
        train_set,label_set,weight_set = get_bagging_data()
        result_set,clf = train_and_predict(CLASSFIER_TO_USE,train_set,label_set,test_set,weight_set)
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
            if(label_set[i] != result_set[i]):
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

        
        one_sum = 0.0
        zero_sum = 0.0
        for i in one_set:
            one_sum += sample_weight_set[i]
        for i in zero_set:
            zero_sum += sample_weight_set[i]
        one_sum *= 2
        zero_sum *= 2
        for i in one_set:
            sample_weight_set[i] /= one_sum
        for i in zero_set:
            sample_weight_set[i] /= zero_sum

        sample_weight_set = normalize(sample_weight_set)
        clf_weight_set.append(math.log(1/beta))
        res = clf.predict(test_set)
        if(TEST_ON_TRAIN_SET):
            check_result(res,real_label_set)
        result_sets.append(res)

    result = calculate_result(result_sets,final_t,weights=clf_weight_set)
    return result

if __name__ == '__main__':
    read_train_and_test_data()
    global train_data,train_label,test_data
    print('data loaded')
    results = []
    pick_test_set = get_test_set()
    pick_test_label_set = None
    # print(train_reviewText[0])
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