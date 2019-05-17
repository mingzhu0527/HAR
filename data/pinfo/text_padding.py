# import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import copy

nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe('sentencizer'))

sent_maxlen = 20
sent_maxnum = 20

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def get_neg_sam(pid, a_start, a_len, neg_sam_num, n):
    l = []
    # neg_sam_num = 4
    inner_sam_num = 3 #int(neg_sam_num/2)
    if a_len <= 1 + inner_sam_num:
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

#pinfo_retrieval
def process_file_pinfo_retrieval(filename, data_type, word_counter, neg_sam_num):
    fh = open(filename, "r")
    big_dict_all = json.load(fh)
    paras = big_dict_all["paras"]
    ques = big_dict_all["ques"]
    a_range_list = big_dict_all["article_range"]
    return process_file(data_type, word_counter, paras, ques, a_range_list, neg_sam_num)



def process_file(data_type, word_counter, paras, ques, a_range_list, neg_sam_num):
    print("Generating {} examples...".format(data_type))
    examples = []
    paras_in_this_article = []
    para_sentence_list = []
    limit = 10
    total = 0
    aIdl = []
    data_para = []
    data_sent = []
    data_query = []
    data_label = []
    snum_l = []
    slen_l = []
    qlen_l = []
    for ind, item in enumerate(a_range_list):
        for i in range(item[1]):
            aIdl.append(ind)
    n = len(paras)
    for i in tqdm(range(n)):
        sent_list = []
        context = paras[i].replace(
                "''", '" ').replace("``", '" ')
        doc = nlp(context)
        sentences = [sent.string.strip() for sent in doc.sents]
        snum_l.append(len(sentences))
        ct_list = []
        for sent in sentences:
            context_tokens = word_tokenize(sent)
            slen = len(context_tokens)
            slen_l.append(slen)
            for token in context_tokens:
                ct_list.append(token)
                word_counter[token] += len(ques[i])
            if slen < sent_maxlen:
                for j in range(sent_maxlen - slen):
                    context_tokens.append("_")
            else:
                context_tokens = context_tokens[:sent_maxlen]
            sent_list.append(context_tokens)
        if len(sent_list) < sent_maxnum:
            for j in range(sent_maxnum - len(sent_list)):
                sent_list.append(["_" for _ in range(sent_maxlen)])
        else:
            sent_list = sent_list[:sent_maxnum]
        para_sentence_list.append(sent_list)
        paras_in_this_article.append(ct_list)
        
    for i in tqdm(range(n)):
        a_start = a_range_list[aIdl[i]][0]
        a_len = a_range_list[aIdl[i]][1]
        for j, que in enumerate(ques[i]):
            question = que.replace("''", '" ').replace("``", '" ').replace("\t", ' ')
            question = que.replace("!!!", '').replace("$$$", '')\
                .replace("@@@", '').replace("^^^", '')\
                .replace("###", '').replace("~~~", '').replace("***", '')
            que_tokens = word_tokenize(question)
            for token in que_tokens:
                word_counter[token] += 1
            qlen = len(que_tokens)
            qlen_l.append(qlen)
            if qlen < sent_maxlen:
                for j in range(sent_maxlen - qlen):
                    que_tokens.append("_")
            else:
                que_tokens = que_tokens[:sent_maxlen]
            question = " ".join(que_tokens)
            data_para.append(" ".join(paras_in_this_article[i]))
            data_sent.append(" ".join([" ".join(x) for x in para_sentence_list[i]]))
            data_query.append(question)
            data_label.append(1)
            neg_sam = get_neg_sam(i - a_start, a_start, a_len, neg_sam_num, n)
            for k in neg_sam:
                data_para.append(" ".join(paras_in_this_article[int(k)]))
                data_sent.append(" ".join([" ".join(x) for x in para_sentence_list[int(k)]]))
                data_query.append(question)
                data_label.append(0)
            total += 1
    print("qlen", sum(qlen_l) / float(len(qlen_l)))
    print("slen", sum(slen_l) / float(len(slen_l)))
    print("snum", sum(snum_l) / float(len(snum_l)))
    print("number of queries", total)
    print("number of examples", len(data_para))
    return data_sent, data_query, data_label

def generateDS(fp, fn, neg_sam_num):
    data_para, data_query, data_label = process_file_pinfo_retrieval(fp, "train", word_counter, neg_sam_num)
    with open(fn, 'w') as outfile:
        for query, para, label in zip(data_query, data_para, data_label):
            outfile.write(str(label) + "\t" + query + "\t" + para + "," + "\n")

word_counter, char_counter = Counter(), Counter()
neg_sam_num = 9
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
# data_sent, data_query, data_label = process_file_pinfo_retrieval(fpdev, "train", word_counter, neg_sam_num)
