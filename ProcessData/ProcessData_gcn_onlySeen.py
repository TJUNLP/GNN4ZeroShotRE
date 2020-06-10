#coding=utf-8

__author__ = 'JIA'
import numpy as np
import pickle, codecs
import json
import re, random, math
from keras.utils.np_utils import to_categorical
from ProcessData import getGraph4Text

def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W


def load_vec_txt(fname, vocab, k=300):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v={}
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
    return w2v,k,W


def load_vec_KGrepresentation(fname, vocab, k):

    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    for line in f.readlines():

        values = line.rstrip('\n').split()
        word = ' '.join(values[:len(values)-100])
        coefs = np.asarray(values[len(values)-100:], dtype='float32')
        w2v[word] = coefs
    f.close()

    W = np.zeros(shape=(vocab.__len__(), k))
    for item in vocab:

        try:
            W[vocab[item]] = w2v[item]
        except BaseException:
            print('the rel is not finded ...', item)

    return k, W


def get_relembed_sim_rank(type_W):

    k = len(type_W)
    W = np.zeros(shape=(k, k))
    RankDict = {}
    for i in range(0, len(type_W)):

        i_j = []
        for j in range(0, len(type_W)):

            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j.append(cos)

        try:
            coefs = np.asarray(i_j, dtype='float32')
            W[i] = coefs
        except BaseException:
            print('the rel is not finded ...', i)

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
        word = ' '.join(values[:len(values)-k])
        coefs = np.asarray(values[len(values)-k:], dtype='float32')
        try:
            W[vocab[word]] = coefs
        except BaseException:
            print('the rel is not finded ...', line)

    f.close()

    return k, W


def load_vec_random(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__(), k))

    for i in vocab_c_inx.keys():
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return k, W


def load_vec_Charembed(vocab_c_inx, char_vob,  k=30):

    TYPE = ['location', 'organization', 'person', 'miscellaneous']

    max = 13
    W = {}

    for i, tystr in enumerate(TYPE):
        for ch in tystr:
            if i not in W.keys():
                W[i] = [char_vob[ch]]
            else:
                W[i] += [char_vob[ch]]

        W[i] += [0 for s in range(max-len(tystr))]

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

    c2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

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

    data_s_all=[]
    count = 0
    f = open(file,'r')
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
            count =0
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
        count +=1
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


def get_sentDicts(trainfile,
                  max_context_l, max_e, max_context_m, max_context_r,
                  max_posi, word_vob, target_vob, char_vob, max_c):
    
    tagDict = {}

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

        if data_tag not in tagDict.keys():
            tagDict[data_tag] = []

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


def Create4Classifier_softmax(tagDict_train, shuffle=True, class_num=120):

    labels = []
    data_tag_all = []

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
            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)
            targetvec = np.zeros(class_num)
            data_tag_all.append([tag])

            if class_num == 2:
                targetvec[1] = 1
                labels.append(targetvec)

                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)
                targetvec = np.zeros(class_num)
                keylist = list(tagDict_train.keys())
                ran3 = random.randrange(0, len(keylist))
                if keylist[ran3] == tag:
                    ran3 = (ran3 + 1) % len(keylist)
                data_tag_all.append([keylist[ran3]])
                targetvec[0] = 1
                labels.append(targetvec)

            elif class_num == 120:
                targetvec[tag] = 1
                labels.append(targetvec)

    if shuffle:
        sh = list(zip(data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
                      data_tag_all, labels))
        random.shuffle(sh)
        data_s_all_0[:], data_e1_posi_all_0[:], data_e2_posi_all_0[:], char_s_all_0[:], \
        data_tag_all[:], labels[:] = zip(*sh)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all]

    return pairs, labels


def Create4Classifier_Multi(tagDict_train, shuffle=True, class_num=120):

    labels = []
    data_tag_all = []

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
            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            # targetvec_0 = np.ones((class_num, 1))
            # targetvec_1 = np.zeros((class_num, 1))
            # targetvec_0[tag][0] = 0
            # targetvec_1[tag][0] = 1
            # targetvec = np.concatenate([targetvec_0, targetvec_1], axis=-1)
            # # print(targetvec)
            # targetvec = targetvec.reshape((class_num * 2))
            # # print(targetvec)

            targetvec = np.zeros(class_num)
            targetvec[tag] = 1

            data_tag_all.append([tag])

            labels.append(targetvec)

    if shuffle:
        sh = list(zip(data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
                 data_tag_all, labels))
        random.shuffle(sh)
        data_s_all_0[:], data_e1_posi_all_0[:], data_e2_posi_all_0[:], char_s_all_0[:],\
        data_tag_all[:], labels[:] = zip(*sh)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all]

    return pairs, labels


def Create4Classifier_DyMax(tagDict_train, shuffle=True, class_num=120):

    data_tag_all = []
    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    assert len(tagDict_train.keys()) == class_num

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            tag_list_all = [i for i in range(len(tagDict_train.keys()))]
            tag_list_all[0] = tag
            tag_list_all[tag] = 0
            data_tag_all.append(tag_list_all)


    if shuffle:
        sh = list(zip(data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
                 data_tag_all))
        random.shuffle(sh)
        data_s_all_0[:], data_e1_posi_all_0[:], data_e2_posi_all_0[:], char_s_all_0[:],\
        data_tag_all[:]  = zip(*sh)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all]

    return pairs


def Create4Classifier_Random(tagDict_train, shuffle=True, class_num=120):

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []
    data_tag_all_a = []
    data_tag_all_n = []

    assert len(tagDict_train.keys()) == class_num

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            i += 1

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_tag_all_a.append([tag])

            keylist = list(tagDict_train.keys())
            ran3 = random.randrange(0, len(keylist))
            if keylist[ran3] == tag:
                ran3 = (ran3 + 1) % len(keylist)

            data_tag_all_n.append([keylist[ran3]])


    if shuffle:
        sh = list(zip(data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
                      data_tag_all_a, data_tag_all_n))
        random.shuffle(sh)
        data_s_all_0[:], data_e1_posi_all_0[:], data_e2_posi_all_0[:], char_s_all_0[:],\
        data_tag_all_a[:], data_tag_all_n[:] = zip(*sh)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_a, data_tag_all_n]

    return pairs


def get_data(trainfile, testfile, w2v_file, c2v_file, t2v_file, datafile, w2v_k=300, c2v_k=25, t2v_k=100, maxlen=50):

    """
    数据处理的入口函数
    Converts the input files  into the model input formats
    """

    word_vob, word_id2word, target_vob, target_id2word, max_s, target_vob_train = get_word_index([trainfile, testfile])
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
    max_context_m = 30
    max_context_r = 35

    tagDict_1 = get_sentDicts(testfile,
                              max_context_l, max_e, max_context_m, max_context_r,
                              max_posi, word_vob, target_vob, char_vob, max_c)
    tagDict_2 = get_sentDicts(trainfile,
                              max_context_l, max_e, max_context_m, max_context_r,
                              max_posi, word_vob, target_vob, char_vob, max_c)
    tagDict = {}
    tagDict.update(tagDict_1)
    tagDict.update(tagDict_2)
    assert len(tagDict.keys()) == len(tagDict_1.keys()) + len(tagDict_2.keys())
    print('tagDict_train len', len(tagDict))

    tagDict_train = {}
    tagDict_dev = {}
    tagDict_test = {}

    for key in tagDict.keys():
        sp = len(tagDict[key]) // 10 * 9
        tagDict_train[key] = tagDict[key][:(sp // 10 * 8)]
        tagDict_dev[key] = tagDict[key][(sp // 10 * 8):sp]
        tagDict_test[key] = tagDict[key][sp:]

    print('tagDict_train len', len(tagDict_train),
          'tagDict_dev len', len(tagDict_dev),
          'tagDict_test len', len(tagDict_test))

    print(datafile, "dataset created!")
    out = open(datafile, 'wb')
    pickle.dump([tagDict_train, tagDict_dev, tagDict_test,
                word_vob, word_id2word, word_W, w2v_k,
                 char_vob, char_id2char, char_W, c2v_k,
                 target_vob, target_id2word,
                 posi_W, posi_k, type_W, type_k,
                max_s, max_posi, max_c], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    class_num = 5
    tag = 3
    targetvec_0 = np.ones((class_num, 1))
    targetvec_1 = np.zeros((class_num, 1))
    targetvec_0[tag][0] = 0
    targetvec_1[tag][0] = 1
    targetvec = np.concatenate([targetvec_0, targetvec_1], axis=-1)
    print(targetvec)
    targetvec = targetvec.reshape((class_num * 2))
    print(targetvec)

    targetvec = np.zeros(class_num)
    targetvec[tag] = 1
    print(targetvec)

    item = np.argmax(targetvec)

    class_max = np.max(targetvec)
    print(item, class_max)






