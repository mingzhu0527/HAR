import json
import re
import numpy as np
import spacy
import json
from tqdm import tqdm
import random
import unicodedata


nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe('sentencizer'))

DigitsMapper = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7','eight': '8', 'nine': '9', 'ten': '10'}

def normal_query(query, document):
    """ normalize digits
    """
    nq = []
    for w in query:
        if w in DigitsMapper and w not in document:
            if DigitsMapper[w] in document:
                w = DigitsMapper[w]
        nq.append(w)
    return nq


def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def token_extend(reg_rules):
    return ' ' + reg_rules.group(0) + ' '

def reform_text(text):
    text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', token_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def get_neg_sam(pid, a_start, a_len, neg_sam_num, n):
    l = []
    # neg_sam_num = 4
    inner_sam_num = 1 #int(neg_sam_num/2)
    if a_len <= 2 + inner_sam_num:
        for k in range(a_len):
            if k != pid:
                l.append(a_start + k)
    else:
        p = np.concatenate((np.arange(0, pid), np.arange(pid + 1, a_len)), axis = 0)
        rand_inner = np.random.choice(p, inner_sam_num, replace = False)
        for k in rand_inner:
            l.append(a_start + k)
            # print("get_neg_sam", pid, a_start + k)
    p = np.concatenate((np.arange(0, a_start), np.arange(a_start + a_len, n)), axis = 0)
    replace = False
    if neg_sam_num - len(l) > len(p):
        replace = True
    rand_outer = np.random.choice(p, neg_sam_num - len(l), replace = replace)
    rand_indices = np.concatenate((l, rand_outer), axis = 0)
    return rand_indices

def process_file_pinfo(filename, neg_sam_num):
    fh = open(filename, "r")
    big_dict_all = json.load(fh)
    paras = big_dict_all["paras"]
    ques = big_dict_all["ques"]
    a_range_list = big_dict_all["article_range"]
    return process_file(paras, ques, a_range_list, neg_sam_num)


def process_file(paras, ques, a_range_list, neg_sam_num):
    aIdl = []
    data_left = []
    data_right = []
    data_label = []
    for ind, item in enumerate(a_range_list):
        for i in range(item[1]):
            aIdl.append(ind)
    n = len(paras)
    for i in range(n):
        paras[i] = paras[i].replace("''", '" ').replace("``", '" ').replace("\t", ' ')
#         paras[i] = re.sub(r'[^\w\s]','',paras[i])
        context_tokens = word_tokenize(paras[i])
        paras[i] = " ".join(context_tokens)
    for i in range(n):
        a_start = a_range_list[aIdl[i]][0]
        a_len = a_range_list[aIdl[i]][1]
        for j, que in enumerate(ques[i]):
            question = que.replace("''", '" ').replace("``", '" ').replace("\t", ' ')
            question = que.replace("!!!", '').replace("$$$", '')\
                .replace("@@@", '').replace("^^^", '')\
                .replace("###", '').replace("~~~", '').replace("***", '')
#             question = re.sub(r'[^\w\s]','',question)
            question_tokens = word_tokenize(question)
            question = " ".join(question_tokens)
            data_left.append(paras[i])
            data_right.append(question)
            data_label.append(1)
            neg_sam = get_neg_sam(i - a_start, a_start, a_len, neg_sam_num, n)
            for k in neg_sam:
                data_left.append(paras[int(k)])
                data_right.append(question)
                data_label.append(0)
    return data_left, data_right, data_label

def process_file_equal_neg(paras, ques, a_range_list, neg_sam_num):
    aIdl = []
    data_left = []
    data_right = []
    data_label = []
    for ind, item in enumerate(a_range_list):
        for i in range(item[1]):
            aIdl.append(ind)
    n = len(paras)
    for i in range(n):
        paras[i] = paras[i].replace("''", '" ').replace("``", '" ').replace("\t", ' ')
#         paras[i] = re.sub(r'[^\w\s]','',paras[i])
        context_tokens = word_tokenize(paras[i])
        paras[i] = " ".join(context_tokens)
    for i in range(n):
        a_start = a_range_list[aIdl[i]][0]
        a_len = a_range_list[aIdl[i]][1]
        for j, que in enumerate(ques[i]):
            question = que.replace("''", '" ').replace("``", '" ').replace("\t", ' ')
            question = que.replace("!!!", '').replace("$$$", '')\
                .replace("@@@", '').replace("^^^", '')\
                .replace("###", '').replace("~~~", '').replace("***", '')
#             question = re.sub(r'[^\w\s]','',question)
            question_tokens = word_tokenize(question)
            question = " ".join(question_tokens)
            neg_sam = get_neg_sam(i - a_start, a_start, a_len, neg_sam_num, n)
            for k in neg_sam:
                data_left.append(paras[i])
                data_right.append(question)
                data_label.append(1)
                data_left.append(paras[int(k)])
                data_right.append(question)
                data_label.append(0)
    return data_left, data_right, data_label

def generateDS(fp, fn, neg_sam_num):
    data_para, data_query, data_label = process_file_pinfo(fp, neg_sam_num)
    with open(fn, 'w') as outfile:
        for query, para, label in zip(data_query, data_para, data_label):
            outfile.write(str(label) + "\t" + query + "\t" + para + "," + "\n")
            
fptrain = "./pinfo_data/all_train.json"
fptest = "./pinfo_data/all_test.json"
fpdev = "./pinfo_data/all_val.json"
ftrain = "pinfo-mz-train.txt"
ftest = "pinfo-mz-test.txt"
fdev = "pinfo-mz-dev.txt"
# ftrain = "pinfo-mz-train.txt"
# ftest = "pinfo-mz-test.txt"
# fdev = "pinfo-mz-dev.txt"
source_files = [fptrain, fptest, fpdev]
dest_files = [ftrain, ftest, fdev]
neg_sam_num = 9
for src, dest in zip(source_files, dest_files):
    generateDS(src, dest, neg_sam_num)