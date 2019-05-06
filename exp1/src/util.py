import codecs
import re

MAINDIR = ".\\trec06c-utf8\\data_cut\\"

MAX_FILE_ID = 300
CONCERNED_SIGN = ['╲', '□','㎡','┻','$','￥','┣','╋','VIP','±','【','】','>','<','[',']','゜','︵','↙']
SPECIAL_KEY = ['night','morning','deep_night','afternoon','noon','phone','http','priorty3','priorty2','priorty1']

def readfile(total_id):
    folderId = total_id / 300
    fileId = total_id % 300
    folderId = str(int(folderId))
    while(len(folderId) < 3):
        folderId = "0" + folderId
    fileId = str(int(fileId))
    while(len(fileId) < 3):
        fileId = "0" + fileId
    f = open(MAINDIR + folderId + "\\" + fileId,'r',encoding="utf-8")
    content = f.read()
    res = []

    # pattern = re.compile(r'Received: from ([0-9a-zA-Z\\.]+)[com,cn,net]'))
    pattern = re.compile(r'From: .* <(.*@.*)>', re.I)
    where = pattern.findall(content)
    # print(where)

    # print(where)
    newwhere = []
    if len(where) > 0:
        where = where[0]
        where = where.split('.')
        for w in where:
            n = w.split('@')
            for nn in n:
                newwhere.append(nn)
        for w in newwhere:
            # print(w)
            if w == ' ' or w == '':
                continue
            res.append(w)

    num_pattern = re.compile(r'[0-9]{3}[0-9]*')

    time_pattern = re.compile(r'[0-9]{4} ([0-9][0-9]):[0-9][0-9]:[0-9][0-9]')
    send_time = time_pattern.findall(content)
    if len(send_time) > 0:
        # print(send_time)
        send_time = int(send_time[0])
        if(send_time > 18):
            res.append('night')
        elif(send_time >15):
            res.append('afternoon')
        elif(send_time > 11):
            res.append('noon')
        elif(send_time > 6):
            res.append('morning')
        elif(send_time >= 0):
            res.append('deep_night')

    x_priority_pattern = re.compile(r'X-Priority: ([0-9])')
    x_priority= x_priority_pattern.findall(content)
    if len(x_priority) > 0:
        # print('priorty'+x_priority[0])
        res.append('priorty'+x_priority[0])

    x_mailer_pattern = re.compile(r'X-Mailer: ([a-zA-Z0-9]+)')
    x_mailer= x_mailer_pattern.findall(content)
    if len(x_mailer) > 0:
        # print(x_mailer)
        res.append(x_mailer[0])
        if x_mailer[0] not in SPECIAL_KEY:
            # print(x_mailer[0])
            SPECIAL_KEY.append(x_mailer[0])

    content = content.replace("\n"," ")
    words = content.split(" ")
    flag = True

    for word in words:
        if len(word) <= 0:
            continue
        if(flag):
            if word[0] >= '\u4e00' and word[0] <= '\u9fff':
                flag = False
            else:
                continue
        
        if(word == 'http' or word == 'mail' or word == 'net' or word == 'com' or word == 'qq'):
            res.append('http')
        if(num_pattern.match(word)):
            # print(word)
            res.append('phone')
        if word[0] >= '\u4e00' and word[0] <= '\u9fff':
            res.append(word)
        elif len(word) == 1:
            if word in CONCERNED_SIGN:
                res.append(word)
    f.close()
    res = list(set(res))
    return res

def initLabel():
    f = open(MAINDIR + '..\\label\\index','r')
    lines = f.readlines()
    label = []
    spam_cnt = 0
    ham_cnt = 0
    for line in lines:
        if line[0] == 's':
            label.append(True)
            spam_cnt += 1
        else:
            label.append(False)
            ham_cnt += 1
    return label,spam_cnt,ham_cnt