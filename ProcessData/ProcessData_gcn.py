# coding=utf-8

__author__ = 'JIA'

import numpy as np
import pickle, codecs
import json
import re, random, math
import keras
from keras.utils.np_utils import to_categorical


def load_vec_pkl(fname, vocab, k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname, 'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v, k, W


def load_vec_txt(fname, vocab, k=300):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f.readlines():
        values = line.rstrip('\n').split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        lower_word = word.lower()
        if not w2v.__contains__(lower_word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken += 1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[lower_word]

    print('UnKnown tokens in w2v', unknowtoken)
    return w2v, k, W


def load_vec_KGrepresentation(fname, vocab, k):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    for line in f.readlines():
        values = line.rstrip('\n').split()
        word = ' '.join(values[:len(values) - 100])
        coefs = np.asarray(values[len(values) - 100:], dtype='float32')
        w2v[word] = coefs
    f.close()

    W = np.zeros(shape=(vocab.__len__(), k))
    for item in vocab:

        try:
            W[vocab[item]] = w2v[item]
        except BaseException:
            print('the rel is not finded ...', item)

    return k, W


def get_rel_sim_rank(type_W):
    RankDict = {}
    for i in range(0, len(type_W)):

        i_j = {}
        mw = 0
        maxs = 0
        for j in range(0, len(type_W)):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos
            if cos > maxs:
                maxs = cos
                mw = j

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        # print('------------', target_id2word[i])
        # for si, s in enumerate(ijlist):
        #     if si > 10:
        #         break
        #     print(i, si, s[0], target_id2word[s[0]], s[1])

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    return RankDict


def get_rel_sim_rank_onlytest(type_W, target_vob, target_vob_train):
    devlist = [7, 14, 25, 39, 43, 55, 60, 72, 83]
    testlist = list(set(target_vob.values()) - set(target_vob_train.values())) + devlist
    trainlist = list(set(target_vob_train.values()) - set(devlist))
    # print(trainlist)
    assert len(testlist) == (24 + 9)
    assert len(trainlist) == (120 - 9 - 24)

    RankDict = {}
    for ii, i in enumerate(testlist):

        i_j = {}

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        # RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]
        RankDict[i] = ijdict

    for k in RankDict.keys():
        print(k, target_id2word[k])
        print(RankDict[k])

    # tmpdict = {}
    # for r in RankDict:
    #     print(r, RankDict[r][0])
    #     if RankDict[r][0] not in tmpdict:
    #         tmpdict[RankDict[r][0]] = []
    #     tmpdict[RankDict[r][0]].append(r)
    #
    # print(len(RankDict.keys()))
    # print(tmpdict)

    return RankDict


def CombineLabel_by_relembed_sim_rank(type_W, target_vob, target_vob_train):
    k = len(type_W)
    W = np.zeros(shape=(k, k), dtype='float32')

    RankDict = {}
    for i in target_vob_train.values():

        i_j = []
        testlist = list(set(target_vob.values()) - set(target_vob_train.values()))
        assert len(testlist) == 24
        for j in testlist:
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            W[i][j] = cos

    while W.max() > 0:
        maxposi = np.unravel_index(np.argmax(W), W.shape)
        # maxposi = np.argwhere(W.max() == W)
        RankDict[maxposi[1]] = maxposi[0]

        print('888', maxposi[1], RankDict[maxposi[1]], W.max())

        # W[maxposi[0], :] *= 0
        W[:, maxposi[1]] *= 0

    assert len(RankDict) == 24
    print(RankDict)

    # ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

    # print('------------', target_id2word[i])
    # for si, s in enumerate(ijlist):
    #     if si > 10:
    #         break
    #     print(target_id2word[s[0]], s[1])

    # ijlist = dict(ijlist)
    # ijlist = list(ijlist.keys())
    # RankDict[i] = ijlist

    # RankDict[i] = list(i_j.values())

    # rank = {}
    # for p, tup in enumerate(ijlist):
    #     rank[tup[0]] = p+1
    # RankDict[i] = rank

    # print(RankDict[i])

    return k, W


def load_vec_ClassSimRank(fname, vocab, k=120):
    W = np.zeros(shape=(k, k))

    f = codecs.open(fname, 'r', encoding='utf-8')

    for line in f.readlines():

        values = line.rstrip('\n').split()
        word = ' '.join(values[:len(values) - k])
        coefs = np.asarray(values[len(values) - k:], dtype='float32')
        try:
            W[vocab[word]] = coefs
        except BaseException:
            print('the rel is not finded ...', line)

    f.close()

    return k, W


def load_vec_random(vocab_c_inx, k=30):
    W = np.zeros(shape=(vocab_c_inx.__len__(), k))

    for i in vocab_c_inx.keys():
        W[vocab_c_inx[i]] = np.random.uniform(-1 * math.sqrt(3 / k), math.sqrt(3 / k), k)

    return k, W


def load_vec_Charembed(vocab_c_inx, char_vob, k=30):
    TYPE = ['location', 'organization', 'person', 'miscellaneous']

    max = 13
    W = {}

    for i, tystr in enumerate(TYPE):
        for ch in tystr:
            if i not in W.keys():
                W[i] = [char_vob[ch]]
            else:
                W[i] += [char_vob[ch]]

        W[i] += [0 for s in range(max - len(tystr))]

    return max, W


def load_vec_character(c2vfile, vocab_c_inx, k=50):
    fi = open(c2vfile, 'r')
    c2v = {}
    for line in fi:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        c2v[word] = coefs
    fi.close()

    c2v["**UNK**"] = np.random.uniform(-1 * math.sqrt(3 / k), math.sqrt(3 / k), k)

    W = np.zeros(shape=(vocab_c_inx.__len__() + 1, k))

    for i in vocab_c_inx:
        if not c2v.__contains__(i):
            c2v[i] = c2v["**UNK**"]
            W[vocab_c_inx[i]] = c2v[i]
        else:
            W[vocab_c_inx[i]] = c2v[i]

    return W, k


def load_vec_onehot(k=124):
    vocab_w_inx = [i for i in range(0, k)]

    W = np.zeros(shape=(vocab_w_inx.__len__(), k))

    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.

    return k, W


def make_idx_character_index(file, max_s, max_c, source_vob):
    data_s_all = []
    count = 0
    f = open(file, 'r')
    fr = f.readlines()

    data_w = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)

            for inum in range(0, num):
                data_tmp = []
                for i in range(0, max_c):
                    data_tmp.append(0)
                data_w.append(data_tmp)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_w)

            data_w = []
            count = 0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]

        for chr in range(0, min(word.__len__(), max_c)):
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])

        num = max_c - word.__len__()
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count += 1
        data_w.append(data_c)

    f.close()
    return data_s_all


def get_word_index(files):
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    target_vob_train = {}
    count = 1
    tarcount = 0

    max_s = 0

    if not source_vob.__contains__("**PlaceHolder**"):
        source_vob["**PlaceHolder**"] = count
        sourc_idex_word[count] = "**PlaceHolder**"
        count += 1
    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    for testf in files:

        f = codecs.open(testf, 'r', encoding='utf-8')
        for line in f.readlines():

            jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
            sent = jline['sent']
            rel = jline['rel']
            words = sent.split(' ')
            for word in words:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1
            if not target_vob.__contains__(rel):
                target_vob[rel] = tarcount
                target_idex_word[tarcount] = rel
                tarcount += 1

            if testf == files[0]:
                if not target_vob_train.__contains__(rel):
                    target_vob_train[rel] = target_vob[rel]

            max_s = max(max_s, len(words))

        f.close()

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, target_vob_train


def get_Character_index(files):
    source_vob = {}
    sourc_idex_word = {}
    max_c = 0
    count = 1

    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**Placeholder**"):
        source_vob["**Placeholder**"] = 1
        sourc_idex_word[1] = "**Placeholder**"
        count += 1

    for file in files:

        f = codecs.open(file, 'r', encoding='utf-8')
        for line in f.readlines():
            jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
            sent = jline['sent']
            rel = jline['rel']
            words = sent.split(' ')

            for word in words:
                max_c = max(max_c, len(word))
                for character in word:
                    if not source_vob.__contains__(character):
                        source_vob[character] = count
                        sourc_idex_word[count] = character
                        count += 1

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_c


def get_prototypes_byques(target_vob, word_vob):
    quesfile = './data/WikiReading/WikiReading.quenstion.txt'

    tagDict = {}

    f = codecs.open(quesfile, 'r', encoding='utf-8')
    lines = f.readlines()
    max_s = 245
    for si, line in enumerate(lines):
        jline = line.rstrip('\n').split(' \t ')
        sent = jline[1].split(' ')
        rel = target_vob[jline[0]]

        if rel not in tagDict:
            tagDict[rel] = []

        data_s = []
        for ww in sent[0:min(len(sent), max_s)]:
            if ww not in word_vob:
                # word_vob[ww] = word_vob['**UNK**']
                data_s.append(word_vob['**UNK**'])
            else:
                data_s.append(word_vob[ww])
        data_s = data_s + [0] * max(0, max_s - len(sent))

        tagDict[rel] = data_s

    print(len(tagDict))

    f.close()

    return max_s, tagDict


def get_sentDicts_neg(trainfile, max_s, max_posi, word_vob, char_vob, max_c):
    tagDict = {}

    tagDict[-1] = []

    f = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f.readlines()

    for si, line in enumerate(lines):
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_posi'][0]
        e1_r = jline['e1_posi'][1]
        e2_l = jline['e2_posi'][0]
        e2_r = jline['e2_posi'][1]

        max_long = max(e1_r, e2_r)
        if len(sent) > max_s and max_long > max_s:
            continue

        data_tag = -1

        # word_vob['____'] = len(word_vob)+1

        data_s = []
        for ww in sent[0:min(len(sent), max_s)]:
            if ww not in word_vob:
                data_s.append(word_vob['**UNK**'])
            else:
                data_s.append(word_vob[ww])
        data_s = data_s + [0] * max(0, max_s - len(sent))

        list_left = [min(i, max_posi) for i in range(1, e1_l + 1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(e1_l, e1_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - e1_r + 1)]
        data_e1_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        list_left = [min(i, max_posi) for i in range(1, e2_l + 1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(e2_l, e2_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - e2_r + 1)]
        data_e2_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        char_s = []
        for wi in range(0, min(len(sent), max_s)):
            word = sent[wi]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])
            data_c = data_c + [0] * max(max_c - word.__len__(), 0)
            char_s.append(data_c)
        char_s = char_s + [[0] * max_c] * max(0, max_s - len(char_s))

        pairs = [data_s, data_e1_posi, data_e2_posi, char_s]

        if data_tag not in tagDict.keys():
            tagDict[data_tag] = []
            # if prototypes != None and data_tag in prototypes.keys():
            #     tagDict[data_tag].append(prototypes[data_tag][0])

        tagDict[data_tag].append(pairs)

    f.close()

    return tagDict


def loadText(flag, sent, max_s, max_c, word_vob, char_vob):

    data_s = []
    for ww in sent[0:min(len(sent), max_s)]:
        if ww not in word_vob:
            word_vob[ww] = word_vob['**UNK**']
        data_s.append(word_vob[ww])

    char_s = []
    for wi in range(0, min(len(sent), max_s)):
        word = sent[wi]
        data_c = []
        for chr in range(0, min(word.__len__(), max_c)):
            if not char_vob.__contains__(word[chr]):
                data_c.append(char_vob["**UNK**"])
            else:
                data_c.append(char_vob[word[chr]])
        data_c = data_c + [0] * max(max_c - word.__len__(), 0)
        char_s.append(data_c)

    if len(sent) < max_s:
        if flag == 'context_l':
            data_s = [0] * (max_s - len(sent)) + data_s
            char_s = [[0] * max_c] * (max_s - len(sent)) + char_s
        elif flag == 'context_r':
            data_s = data_s + [0] * (max_s - len(sent))
            char_s = char_s + [[0] * max_c] * (max_s - len(sent))
        else:
            data_s = [0] * ((max_s - len(sent))//2) + data_s + [0] * (max_s - len(sent) - (max_s - len(sent))//2)
            char_s = [[0] * max_c] * ((max_s - len(sent))//2) + char_s + [[0] * max_c] * (max_s - len(sent) - (max_s - len(sent))//2)

    return data_s, char_s


def get_sentDicts(trainfile,
                  max_context_l, max_e, max_context_m, max_context_r,
                  max_posi, word_vob, target_vob, char_vob, max_c,
                  needDEV=False, target_vob_4dev=None):
    tagDict = {}
    tagDict_dev = {}

    f = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f.readlines()
    for si, line in enumerate(lines):
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_posi'][0]
        e1_r = jline['e1_posi'][1]
        e2_l = jline['e2_posi'][0]
        e2_r = jline['e2_posi'][1]

        if (e1_r - e1_l) > max_e or (e2_r - e2_l) > max_e:
            continue
        if e2_l < e1_r:
            tmp = e1_l
            e1_l = e2_l
            e2_l = tmp
            tmp = e1_r
            e1_r = e2_r
            e2_r = tmp
        if (e2_l - e1_r) > max_context_m:
            continue

        data_tag = target_vob[rel]

        data_s = [word_vob['**UNK**']]
        data_s = data_s * 6
        data_s_cl, char_s_cl = loadText(flag='context_l', sent=sent[max(0, e1_l - max_context_l):e1_l],
                                        max_s=max_context_l, word_vob=word_vob,
                                        max_c=max_c, char_vob=char_vob)
        data_s_e1, char_s_e1 = loadText(flag='e1', sent=sent[e1_l: e1_r],
                                        max_s=max_e, word_vob=word_vob,
                                        max_c=max_c, char_vob=char_vob)
        data_s_cm, char_s_cm = loadText(flag='context_m', sent=sent[e1_r:e2_l],
                                        max_s=max_context_m, word_vob=word_vob,
                                        max_c=max_c, char_vob=char_vob)
        data_s_e2, char_s_e2 = loadText(flag='e2', sent=sent[e2_l: e2_r],
                                        max_s=max_e, word_vob=word_vob,
                                        max_c=max_c, char_vob=char_vob)
        data_s_cr, char_s_cr = loadText(flag='context_r', sent=sent[e2_r: min(len(sent), e2_r + max_context_r)],
                                        max_s=max_context_r, word_vob=word_vob,
                                        max_c=max_c, char_vob=char_vob)
        data_s = data_s + data_s_cl + data_s_e1 + data_s_cm + data_s_e2 + data_s_cr
        char_s = char_s_cl + char_s_e1 + char_s_cm + char_s_e2 + char_s_cr

        list_left = [min(i, max_posi) for i in range(1, max_context_l + 1)]
        list_left.reverse()
        data_e1_posi = list_left + [0 for i in range(0, max_e)] + \
                       [min(i, max_posi) for i in range(1, max_context_m+max_e+max_context_r + 1)]

        list_left = [min(i, max_posi) for i in range(1, max_context_l+max_e+max_context_m + 1)]
        list_left.reverse()
        data_e2_posi = list_left + [0 for i in range(0, max_e)] + \
                       [min(i, max_posi) for i in range(1, max_context_r + 1)]

        pairs = [data_s, data_e1_posi, data_e2_posi, char_s]

        if needDEV is True and rel in target_vob_4dev.keys():
            if data_tag not in tagDict_dev.keys():
                tagDict_dev[data_tag] = []
            tagDict_dev[data_tag].append(pairs)

        else:
            if data_tag not in tagDict.keys():
                tagDict[data_tag] = []
            tagDict[data_tag].append(pairs)

    f.close()

    return tagDict, tagDict_dev


def CreatePairs(tagDict_train, istest=False):
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            labels.append(0)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_tag_all]

    return pairs, labels


def CreateTriplet(tagDict_train, target_vob=None, istest=False):
    categorical_labels = None
    if target_vob != None:
        categorical_labels = to_categorical(list(target_vob.values()), num_classes=None)
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1
            if target_vob != None:
                labels.append(categorical_labels[tag])
            else:
                labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_sample(tagDict_train, target_vob=None, sample_n=2000):
    categorical_labels = None
    if target_vob != None:
        categorical_labels = to_categorical(list(target_vob.values()), num_classes=None)
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc0 = random.randrange(0, len(sents))
        inc1 = random.randrange(0, len(sents))
        if inc1 == inc0:
            inc1 = (inc1 + 1) % len(sents)
        i = 0
        count = 0
        while i < len(sents):
            if count >= sample_n:
                break

            p0 = (inc0 + i) % len(sents)
            p1 = (inc1 + i) % len(sents)

            i += 1
            if target_vob != None:
                labels.append(categorical_labels[tag])
            else:
                labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            count += 1
            if i == len(sents):
                i = 0
                inc0 = (inc0 + random.randrange(1, len(sents))) % len(sents)
                inc1 = (inc1 + random.randrange(1, len(sents))) % len(sents)
                if inc1 == inc0:
                    inc1 = (inc1 + 1) % len(sents)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_DirectMAP(tagDict_train, target_vob=None, istest=False):
    labels = []
    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels.append(1)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_tag_all_p.append([tag])

            if target_vob != None:
                keylist = list(target_vob.values())
            else:
                keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)

            data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs, labels


def CreateTriplet_DirectClassify(tagDict_train, target_vob=None, istest=False):
    labels0 = []
    labels1 = []
    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels0.append([1, 0])
            labels1.append([0, 1])

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            data_tag_all_p.append([tag])

            if target_vob == None:

                data_tag_all_n.append([keylist[ran1]])
            else:
                keylist = list(target_vob.values())
                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_tag_all_p, data_tag_all_n]

    return pairs, labels0, labels1


def CreateTriplet_RankClassify(tagDict, relRankDict, istest=False):
    # print(tagDict.keys())
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    if istest == False:
        for tag in relRankDict.keys():

            ranklist = relRankDict[tag]

            for i, itag in enumerate(ranklist):
                if itag in tagDict.keys():
                    sents1 = tagDict[itag]

                    for j, jtag in enumerate(ranklist[(i + 1):]):
                        if jtag in tagDict.keys():
                            sents2 = tagDict[jtag]

                            ran1 = random.randrange(0, len(sents1))
                            data_s, data_e1_posi, data_e2_posi, char_s = sents1[ran1]
                            data_s_all_0.append(data_s)
                            data_e1_posi_all_0.append(data_e1_posi)
                            data_e2_posi_all_0.append(data_e2_posi)
                            char_s_all_0.append(char_s)

                            ran2 = random.randrange(0, len(sents2))
                            data_s, data_e1_posi, data_e2_posi, char_s = sents2[ran2]
                            data_s_all_1.append(data_s)
                            data_e1_posi_all_1.append(data_e1_posi)
                            data_e2_posi_all_1.append(data_e2_posi)
                            char_s_all_1.append(char_s)

                            data_tag_all.append([tag])

    else:
        for tag in tagDict.keys():

            sents1 = tagDict[tag]

            for sent1 in sents1:

                for j, jtag in enumerate(tagDict.keys()):

                    if jtag == tag:
                        continue

                    sents2 = tagDict[jtag]

                    data_s, data_e1_posi, data_e2_posi, char_s = sent1
                    data_s_all_0.append(data_s)
                    data_e1_posi_all_0.append(data_e1_posi)
                    data_e2_posi_all_0.append(data_e2_posi)
                    char_s_all_0.append(char_s)

                    ran2 = random.randrange(0, len(sents2))
                    data_s, data_e1_posi, data_e2_posi, char_s = sents2[ran2]
                    data_s_all_1.append(data_s)
                    data_e1_posi_all_1.append(data_e1_posi)
                    data_e2_posi_all_1.append(data_e2_posi)
                    char_s_all_1.append(char_s)

                    data_tag_all.append([tag])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_tag_all]

    return pairs


def CreateTriplet_RankClassify2(tagDict, relRankDict, target_vob_train=None, istest=False):
    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    if istest == False:

        for tag in tagDict.keys():
            sents = tagDict[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 3)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 3):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

    else:
        for tag in tagDict.keys():
            sents = tagDict[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs


def CreateTriplet_RankClassify3(tagDict, relRankDict, target_vob_train=None, istest=False):
    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    if istest == False:

        for tag in tagDict.keys():
            sents = tagDict[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                span = len(ranklist) // 2
                ran1 = random.randrange(0, span)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + span):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

    else:
        for tag in tagDict.keys():
            sents = tagDict[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs


def CreateTriplet_RankClassify4(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    RankDict = {}
    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}
        testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
        assert len(testlist) == (24 + 9)

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    if istest == False:

        for tag in tagDict_train.keys():
            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs


'''
def CreateTriplet_RankClassify42(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):

    # version 3


    RankDict = {}

    trainlist = list(tagDict_train.keys()) + list(tagDict_dev.keys())
    assert len(trainlist) == 96

    for ii, i in enumerate(trainlist):

        i_j = {}
        testlist = list(tagDict_test.keys())
        assert len(testlist) == 24

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    for tagDict in [tagDict_train, tagDict_dev]:

        for i, tag in enumerate(tagDict.keys()):

            sents = tagDict[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                if istest == False and i < (len(sents) * 0.15):
                    i += 1
                    continue
                if istest == True and i >= (len(sents) * 0.15):
                    i += 1
                    break

                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

                data_tag_all_a.append([tag])

                labels.append(tag)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels]

    return pairs
'''

'''
def CreateTriplet_RankClassify42(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):

    version 2


    RankDict = {}

    trainlist = list(tagDict_train.keys()) + list(tagDict_dev.keys())
    assert len(trainlist) == 96

    for ii, i in enumerate(trainlist):

        i_j = {}
        testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
        assert len(testlist) == (24 + 9)

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    if istest == False:

        for tag in tagDict_train.keys():
            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

                data_tag_all_a.append([tag])

                labels.append(tag)

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(1, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

                data_tag_all_a.append([tag])

                labels.append(tag)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels]

    return pairs
'''


def CreateTriplet_RankClassify42(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    # version 1

    RankDict = {}
    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}
        testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
        assert len(testlist) == (24 + 9)

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []
    data_tag_all_n0 = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    if istest == False:

        for tag in tagDict_train.keys():
            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                keylist = list(tagDict_train.keys())
                ran3 = random.randrange(0, len(keylist))
                if keylist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(keylist)

                data_tag_all_n0.append([keylist[ran3]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                keylist = list(tagDict_dev.keys())
                ran3 = random.randrange(0, len(keylist))
                if keylist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(keylist)

                data_tag_all_n0.append([keylist[ran3]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels, data_tag_all_n0]

    return pairs


def CreateTriplet_RankClassify421(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    # version 1

    RankDict = {}
    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}
        testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
        assert len(testlist) == (24 + 9)

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []
    data_tag_all_n0 = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    if istest == False:

        for tag in tagDict_train.keys():
            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                keylist = list(ranklist[(ran1 + 1):])

                ran2 = random.randrange(0, len(keylist))

                data_tag_all_n.append([keylist[ran2]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                keylist = list(tagDict_train.keys())
                ran3 = random.randrange(0, len(keylist))
                if keylist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(keylist)

                data_tag_all_n0.append([keylist[ran3]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                data_tag_all_n0.append([tag])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels, data_tag_all_n0]

    return pairs


def CreateTriplet_RankClassify521(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
    assert len(testlist) == (24 + 9)

    Rank_te2tr_Dict = {}
    for ji, j in enumerate(testlist):
        j_i = {}

        for ii, i in enumerate(tagDict_train.keys()):
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            j_i[i] = cos

        ijlist = sorted(j_i.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)

        Rank_te2tr_Dict[j] = list(ijdict.keys())

    tr_in_te_Dict = {}

    top_K = 20

    for ki in list(Rank_te2tr_Dict.keys()):
        for va in Rank_te2tr_Dict[ki][:top_K]:

            if va not in tr_in_te_Dict.keys():
                tr_in_te_Dict[va] = []

            assert ki not in tr_in_te_Dict[va]
            tr_in_te_Dict[va].append(ki)

    Rank_tr2te_Dict = {}

    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}

        for ji, j in enumerate(testlist):
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        Rank_tr2te_Dict[i] = list(ijdict.keys())

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []
    data_tag_all_n0 = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    if istest == False:

        for tag in tagDict_train.keys():

            if tag not in tr_in_te_Dict:
                continue

            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_a.append([tag])

                labels.append(tag)

                ranklist = Rank_tr2te_Dict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)
                now_tag = ranklist[ran1]
                data_tag_all_p.append([now_tag])

                keylist = list(ranklist[(ran1 + 1):])
                ran2 = random.randrange(0, len(keylist))
                data_tag_all_n.append([keylist[ran2]])

                assert now_tag in testlist
                ranklist = Rank_te2tr_Dict[now_tag][:top_K]

                ran3 = random.randrange(0, len(ranklist))
                if ranklist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(ranklist)

                data_tag_all_n0.append([ranklist[ran3]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(Rank_tr2te_Dict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                data_tag_all_n0.append([tag])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels, data_tag_all_n0]

    return pairs


def CreateTriplet_RankClassify621(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
    assert len(testlist) == (24 + 9)

    Rank_te2tr_Dict = {}
    for ji, j in enumerate(testlist):
        j_i = {}

        for ii, i in enumerate(tagDict_train.keys()):
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            j_i[i] = cos

        ijlist = sorted(j_i.items(), key=lambda x: x[1], reverse=True)

        threshold = 0.5
        if ijlist[0][1] < threshold:
            continue
        else:
            ijdict = dict(ijlist)
            Rank_te2tr_Dict[j] = list(ijdict.keys())

    Rank_tr2te_Dict = {}

    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}

        for ji, j in enumerate(testlist):
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        Rank_tr2te_Dict[i] = list(ijdict.keys())

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    data_tag_all_p = []
    data_tag_all_n = []
    data_tag_all_a = []
    data_tag_all_n0 = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    labels = []

    if istest == False:

        for tag in tagDict_train.keys():

            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_a.append([tag])

                labels.append(tag)

                ranklist = Rank_tr2te_Dict[tag]
                whiletag = True
                ran1 = -1
                while whiletag:
                    ran1 = random.randrange(0, len(ranklist) - 1)
                    now_tag = ranklist[ran1]
                    if now_tag in Rank_te2tr_Dict.keys():
                        data_tag_all_p.append([now_tag])
                        whiletag = False
                    else:
                        continue

                keylist = list(ranklist[(ran1 + 1):])
                ran2 = random.randrange(0, len(keylist))
                data_tag_all_n.append([keylist[ran2]])

                keylist = list(tagDict_train.keys())
                ran3 = random.randrange(0, len(keylist))
                if keylist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(keylist)

                data_tag_all_n0.append([keylist[ran3]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(Rank_tr2te_Dict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

                data_tag_all_a.append([tag])

                labels.append(tag)

                data_tag_all_n0.append([tag])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n, data_tag_all_a, labels, data_tag_all_n0]

    return pairs


def CreateTriplet_RankClassify41(tagDict_train, tagDict_dev, tagDict_test, type_W, istest=False):
    RankDict = {}
    for ii, i in enumerate(tagDict_train.keys()):

        i_j = {}
        testlist = list(tagDict_dev.keys()) + list(tagDict_test.keys())
        assert len(testlist) == (24 + 9)

        for ji, j in enumerate(testlist):
            # if i == j:
            #     continue
            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j[j] = cos

        # print(i, mw, maxs)

        ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        ijdict = dict(ijlist)
        # print(ijdict)
        RankDict[i] = list(ijdict.keys())
        # assert i == RankDict[i][0]

    # print(RankDict)
    # print(len(RankDict.keys()), len(RankDict[0]))

    # testlist = list(set(relRankDict.keys()) - set(target_vob_train.values()))
    # assert len(testlist) == 24

    # print(tagDict.keys())

    relRankDict = RankDict

    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    if istest == False:

        for tag in tagDict_train.keys():
            sents = tagDict_train[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                ranklist = relRankDict[tag]
                ran1 = random.randrange(0, len(ranklist) - 1)

                data_tag_all_p.append([ranklist[ran1]])

                data_tag_all_n.append([ranklist[ran1 + 1]])

    else:
        for tag in tagDict_dev.keys():
            sents = tagDict_dev[tag]

            if len(sents) < 2:
                continue
            inc = random.randrange(1, len(sents))
            i = 0
            while i < len(sents):
                p0 = i
                p1 = (inc + i) % len(sents)

                i += 1

                data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all_p.append([tag])

                keylist = list(relRankDict.keys())

                ran1 = random.randrange(0, len(keylist))
                if keylist[ran1] == tag:
                    ran1 = (ran1 + 1) % len(keylist)

                data_tag_all_n.append([keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs


def CreateTriplet_DirectMAP_AL(tagDict, tagDict_train, tagDict_dev, tagDict_test):
    labels = []
    unseen = []
    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    for tag in tagDict.keys():
        sents = tagDict[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels.append(1)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_tag_all_p.append([tag])

            if i % 2 == 0:
                keylist = list(tagDict_test.keys()) + list(tagDict_dev.keys())
            else:
                keylist = list(tagDict_train.keys())

            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)

            data_tag_all_n.append([keylist[ran1]])

            unseentarget = np.zeros(2, dtype='int32')
            if keylist[ran1] not in tagDict_train.keys():
                unseentarget[1] = 1
            else:
                unseentarget[0] = 1
            unseen.append(unseentarget)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs, labels, unseen


def CreateTriplet_withSoftmax(tagDict_train, target_vob=None, istest=False):
    """
    if target_vob != None, unseen classes are used for pos examples
    """

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = np.zeros(2)
            if i % 2 == 0:
                targetvec[0] = 1

                if target_vob == None:
                    data_tag_all.append([keylist[ran1]])
                else:
                    keylist = list(target_vob.values())
                    ran1 = random.randrange(0, len(keylist))
                    if keylist[ran1] == tag:
                        ran1 = (ran1 + 1) % len(keylist)
                    data_tag_all.append([keylist[ran1]])
            else:
                targetvec[1] = 1
                data_tag_all.append([tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_withSoftmax_ques(tagDict_train, prototype_tagDict, target_vob=None, istest=False):
    """
    if target_vob != None, unseen classes are used for pos examples
    """

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = np.zeros(2)
            if i % 2 == 0:
                targetvec[0] = 1

                if target_vob == None:
                    data_tag_all.append(prototype_tagDict[keylist[ran1]])
                else:
                    keylist = list(target_vob.values())
                    ran1 = random.randrange(0, len(keylist))
                    if keylist[ran1] == tag:
                        ran1 = (ran1 + 1) % len(keylist)
                    data_tag_all.append(prototype_tagDict[keylist[ran1]])
            else:
                targetvec[1] = 1
                data_tag_all.append(prototype_tagDict[tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_withMSE(tagDict_train):
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = 0
            if i % 2 == 0:
                targetvec = 0
                data_tag_all.append([keylist[ran1]])

            else:
                targetvec = 1
                data_tag_all.append([tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def get_split_train_dev(target_vob_train):
    rel4dev = {}
    relList = list(target_vob_train.keys())
    i = 0
    while i * 10 + 9 < len(relList):
        nd = random.randint(0, 9)
        k = relList[i * 10 + nd]
        rel4dev[k] = target_vob_train[k]
        i += 1
    # while i * 16 + 15 < len(relList):
    #     nd = random.randint(0, 15)
    #     k = relList[i * 16 + nd]
    #     rel4dev[k] = target_vob_train[k]
    #     i += 1

    return rel4dev


def get_rel_prototypes(file, max_s, max_posi, word_vob, target_vob, char_vob, max_c):
    tagDict_prototypes, _ = get_sentDicts(file, max_s, max_posi, word_vob, target_vob, char_vob, max_c, needDEV=False)
    print('tagDict_prototypes len', len(tagDict_prototypes))

    return tagDict_prototypes


def get_data(trainfile, testfile, w2v_file, t2v_file, datafile, w2v_k=300, c2v_k=25,
             t2v_k=100, maxlen=100):
    """
    数据处理的入口函数
    Converts the input files  into the model input formats
    """

    word_vob, word_id2word, target_vob, target_id2word, max_s, target_vob_train = get_word_index(
        [trainfile, testfile])
    print("source vocab size: ", str(len(word_vob)))
    print("word_id2word size: ", str(len(word_id2word)))
    print("target vocab size: " + str(len(target_vob)))
    print("target_id2word size: " + str(len(target_id2word)))
    if max_s > maxlen:
        max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))

    char_vob, char_id2char, max_c = get_Character_index({trainfile, testfile})
    print("source char size: ", char_vob.__len__())
    max_c = min(max_c, 18)
    print("max_c: ", max_c)

    c2v_k, char_W, = load_vec_random(char_vob, k=c2v_k)
    print('character_W shape:', char_W.shape)

    word_w2v, w2v_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(word_vob)))
    print("source_W  size: " + str(len(word_W)))

    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=t2v_k)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))

    max_posi = 20
    posi_k, posi_W = load_vec_onehot(k=max_posi + 1)
    print('posi_k, posi_W', posi_k, len(posi_W))

    max_context_l = 35
    max_e = 6
    max_context_m = 35
    max_context_r = 35

    tagDict_test, _ = get_sentDicts(testfile,
                                    max_context_l, max_e, max_context_m, max_context_r,
                                    max_posi, word_vob, target_vob, char_vob, max_c)
    print('tagDict_test len', len(tagDict_test))

    target_vob_4dev = get_split_train_dev(target_vob_train)
    print('target_vob len', len(target_vob), 'target_vob_4dev len', len(target_vob_4dev))

    tagDict_train, tagDict_dev = get_sentDicts(trainfile,
                                               max_context_l, max_e, max_context_m, max_context_r,
                                               max_posi, word_vob, target_vob, char_vob, max_c,
                                               needDEV=True, target_vob_4dev=target_vob_4dev)
    print('tagDict_train len', len(tagDict_train), 'tagDict_dev len', len(tagDict_dev))

    print(datafile, "dataset created!")
    out = open(datafile, 'wb')
    pickle.dump([tagDict_train, tagDict_dev, tagDict_test,
                 word_vob, word_id2word, word_W, w2v_k,
                 char_vob, char_id2char, char_W, c2v_k,
                 target_vob, target_id2word,
                 posi_W, posi_k, type_W, type_k,
                 max_s, max_posi, max_c], out, 0)
    out.close()


if __name__ == "__main__":
    print(20 * 2)
