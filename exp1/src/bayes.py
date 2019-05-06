from util import *
import random
import math
import time

bias = -6.25
special_key_weight = 10

class MailList:
    def __init__(self,kk):
        self.label,self.spam_cnt,self.ham_cnt = initLabel()
        self.k = kk
        self.randSetInit()
        print("MailList initialized:")
        print("\ttrainingSet Size:",len(self.trainingCases))
        print("\ttestSet Size:",len(self.testCases))
        print("\ttotal spam:",self.spam_cnt)
        print("\ttotal ham:",self.ham_cnt)
        self.train()
        # self.calculate()
        self.test()

    def calculate(self):
        keys = self.dic.keys()
        # keys = list(keys)
        keys = sorted(keys)
        ave = 0.0
        for k in keys:
            ave += float(self.dic[k][0])/float(self.train_spam_cnt)-float(self.dic[k][1])/self.train_ham_cnt
            if k in SPECIAL_KEY or k == '销售'  or k == '以后' or k == '公司':
                if(self.dic[k][0] + self.dic[k][1] <= 10): continue
                print(k,"\t","delta:",float(self.dic[k][0])/float(self.train_spam_cnt)-float(self.dic[k][1])/self.train_ham_cnt)
        input()

    def randSetInit(self):
        randSet = list(range(0,self.spam_cnt + self.ham_cnt))

        #TA said no random num
        random.shuffle(randSet)
        
        self.trainingCases = randSet[0:int(self.k*0.8*(self.spam_cnt + self.ham_cnt))]
        self.testCases = randSet[-(int)(0.2*(self.spam_cnt + self.ham_cnt)): self.spam_cnt + self.ham_cnt]

    def getTrainingMailCut(self,trainingId):
        # print("id:",self.trainingCases[trainingId])
        return readfile(self.trainingCases[trainingId]),self.label[self.trainingCases[trainingId]]

    def getTestMailCut(self,testId):
        return readfile(self.testCases[testId]),self.label[self.testCases[testId]]

    def getTestSetSize(self):
        return len(self.testCases)

    def getTrainingSetSize(self):
        return len(self.trainingCases)

    def getSpamCnt(self):
        return self.spam_cnt

    def getHamCnt(self):
        return self.ham_cnt

    def getTotalCnt(self):
        return self.ham_cnt + self.spam_cnt

    def train(self):
        print("**********************************training************************************")
        self.train_spam_cnt = 0;
        self.train_ham_cnt = 0;
        self.dic = {}
        cur = 1
        for i in range(0,len(self.trainingCases)):
            if float(i)/float(len(self.trainingCases))*100 > cur:
                print("\t",str(cur) + "% ","keyword count:" + str(len(self.dic.keys())))
                cur+=1
            keyWords,l = self.getTrainingMailCut(i)
            if l:
                self.train_spam_cnt += 1
                for k in keyWords:
                    weight = 1
                    if k in SPECIAL_KEY:
                        weight = special_key_weight
                    if(k in self.dic.keys()):
                        self.dic[k][0] += weight
                    else:
                        self.dic[k] = [weight,0]
            else:
                self.train_ham_cnt += 1
                for k in keyWords:
                    weight = 1
                    if k in SPECIAL_KEY:
                        weight = special_key_weight
                    if(k in self.dic.keys()):
                        self.dic[k][1] += weight
                    else:
                        self.dic[k] = [0,weight]

        # for d in self.dic.keys():
        #     if len(d) > 1 or self.dic[d][0] + self.dic[d][1] < 10:
        #         continue
        #     if(self.dic[d][1] == 0):
        #         print(d)
        #         input()
        #     if self.dic[d][0] == 0:
        #         print(d)
        #         input()
        #     if not (self.dic[d][0] == 0 or self.dic[d][1] == 0):
        #         if(self.dic[d][0]/self.dic[d][1] > 100 or self.dic[d][1]/self.dic[d][0] > 100):
        #             print(d)
        #             input()
            

    def test(self):
        print("**********************************testing************************************")
        correctCnt = 0
        falseCnt = 0
        mailCnt = self.train_spam_cnt+self.train_ham_cnt
        cur = 1
        spamCoCnt = 0
        hamCoCnt = 0
        spamFalCnt = 0
        hamFalCnt = 0
        for i in range(0,len(self.testCases)):
            if float(i)/float(len(self.testCases))*100 > cur:
                print("\t",str(cur) + "%" + " corect count:",correctCnt," false count:",falseCnt,"\tspam corect rate:",float(spamCoCnt)/float(spamFalCnt + spamCoCnt),"\tham corect rate:",float(hamCoCnt)/float(hamFalCnt + hamCoCnt))
                cur+=1
            keyWords,l = self.getTestMailCut(i)
            ps = math.log(1.0*float(self.train_spam_cnt)) - math.log(float(mailCnt))
            ph = math.log(1.0*float(self.train_ham_cnt)) - math.log(float(mailCnt))

            # px = 1.0*float(self.train_ham_cnt)/float(mailCnt)

            for k in keyWords:
                if k in self.dic.keys():
                    # print(k,float(self.dic[k][0] + 1)/float(self.train_spam_cnt +notShown)/(float(self.dic[k][0] + self.dic[k][1] + 2)/float(mailCnt)),self.dic[k][0],self.dic[k][1])
                    if(self.dic[k][0] != 0):
                        ps += math.log(self.dic[k][0]) - math.log(float(self.train_spam_cnt))
                    else:
                        ps += math.log(float(1)) - math.log(float(self.train_spam_cnt + (len(self.dic.keys()))))
                    if(self.dic[k][1] != 0):
                        ph += math.log(self.dic[k][1]) - math.log(float(self.train_ham_cnt))
                    else:
                        ph += math.log(float(1)) - math.log(float(self.train_ham_cnt + (len(self.dic.keys()))))
                    # px *= float(self.dic[k][1] + 1)/float(self.train_ham_cnt +notShown)/(float(self.dic[k][0] + self.dic[k][1] + 2)/float(mailCnt))
                else:
                    ps += math.log(float(1)) - math.log(float(self.train_spam_cnt + (len(self.dic.keys()))))
                    # ph += math.log(float(1)) - math.log(float(self.train_ham_cnt + 2))
                    ph += math.log(float(1)) - math.log(float(self.train_spam_cnt + (len(self.dic.keys()))))

            if(ps >= ph + bias and l) or (ps < ph + bias and not l):
                # if( ps == ph ): 
                #     print(ps,ph)
                #     for k in self.dic.keys():
                #         print("\t",self.dic[k][0],self.dic[k][1])
                #         input()
                correctCnt += 1
                if(l):
                    spamCoCnt += 1
                else:
                    hamCoCnt += 1
            else:
                # print(ps-ph,str(l))
                falseCnt += 1
                if(l):
                    spamFalCnt += 1
                else:
                    hamFalCnt += 1

                # print(self.dic[k][0],self.train_spam_cnt,self.dic[k][0] + self.dic[k][1],mailCnt)
                # print(float(self.dic[k][0])/float(self.train_spam_cnt)/(float(self.dic[k][0] + self.dic[k][1])/float(mailCnt)))
                # print(p)
        self.correctRate = float(correctCnt) / float(falseCnt + correctCnt)

if __name__ == '__main__':

    a = input()
    a = float(a)
    time_start=time.time()
    correct_rate = []
    sum = 0.0
    maxmum = 0.0
    minimum = 1.0
    for i in range (0,5):
        print('time:',i)
        m = MailList(a)
        correct_rate.append(m.correctRate)
        print(m.correctRate)
        sum += m.correctRate
        if(m.correctRate > maxmum):
            maxmum = m.correctRate
        if(m.correctRate < minimum):
            minimum = m.correctRate

    print('\n\n------------------------result------------------------')
    print('result:',correct_rate)
    print('average:',str(sum/float(5)*100)[0:8] + "%")
    print('max:',str(maxmum*100)[0:8] + "%")
    print('min:',str(minimum*100)[0:8] + "%")
    print('info: testCases:','20%','trainingCases:',str(a*0.8*100)[0:8] + '%')
    time_end=time.time()
    print('total time using:',str(int(time_end-time_start)) + 's')
    print('weight:',special_key_weight)
    print('bias:',bias)
