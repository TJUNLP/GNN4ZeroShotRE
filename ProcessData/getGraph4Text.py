
import codecs, json


def getLens(files):
    count = 0
    max_context_l = 0
    max_context_m = 0
    max_context_r = 0
    max_e_1 = 1
    max_e_2 = 1
    e1_dict = {}
    e2_dict = {}
    cl_dict = {}
    cm_dict = {}
    cr_dict = {}

    for testf in files:

        f = codecs.open(testf, 'r', encoding='utf-8')
        for line in f.readlines():
            count += 1
            jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
            sent = jline['sent'].split(' ')
            rel = jline['rel']
            e1_l = jline['e1_posi'][0]
            e1_r = jline['e1_posi'][1]
            e2_l = jline['e2_posi'][0]
            e2_r = jline['e2_posi'][1]

            max_e_1 = max(max_e_1, (e1_r - e1_l))
            if (e1_r - e1_l) not in e1_dict.keys():
                e1_dict[(e1_r - e1_l)] = 1/225060
            else:
                e1_dict[(e1_r - e1_l)] += 1/225060
            max_e_2 = max(max_e_2, (e2_r - e2_l))
            if (e2_r - e2_l) not in e2_dict:
                e2_dict[(e2_r - e2_l)] = 1/225060
            else:
                e2_dict[(e2_r - e2_l)] += 1/225060
            max_context_l = max(max_context_l, e1_l)
            if e1_l not in cl_dict.keys():
                cl_dict[e1_l] = 1/225060
            else:
                cl_dict[e1_l] += 1/225060

            max_context_m = max(max_context_m, (e2_l - e1_r))
            if (e2_l - e1_r) not in cm_dict.keys():
                cm_dict[(e2_l - e1_r)] = 1/225060
            else:
                cm_dict[(e2_l - e1_r)] += 1/225060
            # if (e2_l - e1_r) <=0:
            #     print(sent)
            #     count -= 1

            max_context_r = max(max_context_r, (len(sent) - e2_r))
            if (len(sent) - e2_r) not in cr_dict.keys():
                cr_dict[(len(sent) - e2_r)] = 1/225060
            else:
                cr_dict[(len(sent) - e2_r)] += 1/225060

        f.close()

    e1_dict = sorted(e1_dict.items(), key=lambda x: x[0])
    print(e1_dict)
    e2_dict = sorted(e2_dict.items(), key=lambda x: x[0])
    print(e2_dict)
    print(count)
    cl_dict = sorted(cl_dict.items(), key=lambda x: x[0])
    cm_dict = sorted(cm_dict.items(), key=lambda x: x[0])
    print(cm_dict)
    cr_dict = sorted(cr_dict.items(), key=lambda x: x[0])
    cl_add = 0
    for i in cm_dict:
        if i[0] <= 0:
            continue
        cl_add += i[1]
        print(i[0], cl_add)

    # !!!!!!!!!!!!!!!!!!
    # result: use the length of entity is 6
    # use the length of context_l or _r _m is 35
    max_context_l = 35
    max_e_1 = 6
    max_context_m = 35
    max_e_2 = 6
    max_context_r = 35

    return (max_context_l, max_e_1, max_context_m, max_e_2, max_context_r)


def GetGraph(max_context_l=35, max_e_1=6, max_context_m=35, max_e_2=6, max_context_r=35):
    graph_dict = {}
    '''
    define:
        node-0: root
        node-1: context_l overview
        node-2: e_1 overview
        node-3: context_m overview
        node-4: e_2 overview
        node-5: context_r overview
        node-6~40: context_l word
        node-41~46: e_1 word
        node-47~81: context_m word
        node-82~87: e_2 word
        node-88~122: context_r word
    '''
    graph_dict[0] = [1, 1, 1, 1, 1, 1] + [0 for i in range(6, 123)]
    # print(len(graph_dict[0]), graph_dict[0])
    graph_dict[1] = [1, 1, 1, 0, 0, 0] + [1 for i in range(6, 41)] + [0 for i in range(41, 123)]
    graph_dict[2] = [1, 1, 1, 1, 0, 0] + [0 for i in range(6, 41)] + \
                    [1 for i in range(41, 47)] + [0 for i in range(47, 123)]
    graph_dict[3] = [1, 0, 1, 1, 1, 0] + [0 for i in range(6, 47)] + [1 for i in range(47, 82)] + \
                    [0 for i in range(82, 123)]
    graph_dict[4] = [1, 0, 0, 1, 1, 1] + [0 for i in range(6, 82)] + [1 for i in range(82, 88)] + \
                    [0 for i in range(88, 123)]
    graph_dict[5] = [1, 0, 0, 0, 1, 1] + [0 for i in range(6, 88)] + [1 for i in range(88, 123)]
    graph_dict[6] = [0, 1, 0, 0, 0, 0] + [1, 1] + [0 for i in range(8, 123)]

    for i in range(7, 41):
        graph_dict[i] = [0, 1, 0, 0, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i+2, 123)]

    for i in range(41, 47):
        graph_dict[i] = [0, 0, 1, 0, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i+2, 123)]

    for i in range(47, 82):
        graph_dict[i] = [0, 0, 0, 1, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)]

    for i in range(82, 88):
        graph_dict[i] = [0, 0, 0, 0, 1, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)]

    for i in range(88, 122):
        graph_dict[i] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)]

    graph_dict[122] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, 121)] + [1, 1]

    return graph_dict


def GetGraph_withOneTag(max_context_l=35, max_e_1=6, max_context_m=35, max_e_2=6, max_context_r=35):
    graph_dict = {}
    '''
    define:
        node-0: root
        node-1: context_l overview
        node-2: e_1 overview
        node-3: context_m overview
        node-4: e_2 overview
        node-5: context_r overview
        node-6~40: context_l word
        node-41~46: e_1 word
        node-47~81: context_m word
        node-82~87: e_2 word
        node-88~122: context_r word
        node-123: tag
    '''
    graph_dict[0] = [1, 1, 1, 1, 1, 1] + [0 for i in range(6, 123)] + [1]
    # print(len(graph_dict[0]), graph_dict[0])
    graph_dict[1] = [1, 1, 1, 0, 0, 0] + [1 for i in range(6, 41)] + [0 for i in range(41, 123)] + [1]
    graph_dict[2] = [1, 1, 1, 1, 0, 0] + [0 for i in range(6, 41)] + \
                    [1 for i in range(41, 47)] + [0 for i in range(47, 123)] + [1]
    graph_dict[3] = [1, 0, 1, 1, 1, 0] + [0 for i in range(6, 47)] + [1 for i in range(47, 82)] + \
                    [0 for i in range(82, 123)] + [1]
    graph_dict[4] = [1, 0, 0, 1, 1, 1] + [0 for i in range(6, 82)] + [1 for i in range(82, 88)] + \
                    [0 for i in range(88, 123)] + [1]
    graph_dict[5] = [1, 0, 0, 0, 1, 1] + [0 for i in range(6, 88)] + [1 for i in range(88, 123)] + [1]
    graph_dict[6] = [0, 1, 0, 0, 0, 0] + [1, 1] + [0 for i in range(8, 123)] + [1]

    for i in range(7, 41):
        graph_dict[i] = [0, 1, 0, 0, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i+2, 123)] + [1]

    for i in range(41, 47):
        graph_dict[i] = [0, 0, 1, 0, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i+2, 123)] + [1]

    for i in range(47, 82):
        graph_dict[i] = [0, 0, 0, 1, 0, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)] + [1]

    for i in range(82, 88):
        graph_dict[i] = [0, 0, 0, 0, 1, 0] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)] + [1]

    for i in range(88, 122):
        graph_dict[i] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, i-1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, 123)] + [1]

    graph_dict[122] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, 121)] + [1, 1] + [1]

    graph_dict[123] = [1 for j in range(0, 124)]

    return graph_dict


def GetGraph_gat(max_s, pad_left, e1_l, e1_r, e2_l, e2_r):
    graph_dict = {}
    cl_len = pad_left + e1_l
    e1_len = e1_r - e1_l
    cm_len = e2_l - e1_r
    e2_len = e2_r - e2_l
    cr_len = max_s - e2_r - pad_left
    '''
    define:
        node-0: root
        node-1: context_l overview
        node-2: e_1 overview
        node-3: context_m overview
        node-4: e_2 overview
        node-5: context_r overview
        node-6~40: context_l word
        node-41~46: e_1 word
        node-47~81: context_m word
        node-82~87: e_2 word
        node-88~122: context_r word
    '''
    graph_dict[0] = [1, 1, 1, 1, 1, 1] + [0 for i in range(0, max_s)]
    # print(len(graph_dict[0]), graph_dict[0])
    graph_dict[1] = [1, 1, 1, 0, 0, 0] + [1 for i in range(0, cl_len)] + \
                    [0 for i in range(0, max_s-cl_len)]

    graph_dict[2] = [1, 1, 1, 1, 0, 0] + [0 for i in range(0, cl_len)] + \
                    [1 for i in range(0, e1_len)] + [0 for i in range(0, (cm_len+e2_len+cr_len))]

    graph_dict[3] = [1, 0, 1, 1, 1, 0] + [0 for i in range(0, (cl_len+e1_len))] + \
                    [1 for i in range(0, cm_len)] + [0 for i in range(0, (e2_len+cr_len))]

    graph_dict[4] = [1, 0, 0, 1, 1, 1] + [0 for i in range(0, (cl_len+e1_len+cm_len))] + \
                    [1 for i in range(0, e2_len)] + [0 for i in range(0, cr_len)]

    graph_dict[5] = [1, 0, 0, 0, 1, 1] + [0 for i in range(0, max_s-cr_len)] + \
                    [1 for i in range(0, cr_len)]

    graph_dict[6] = [0, 1, 0, 0, 0, 0] + [1, 1] + [0 for i in range(0, max_s-2)]

    for i in range(7, 6+cl_len):
        graph_dict[i] = [0, 1, 0, 0, 0, 0] + [0 for j in range(6, i - 1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, max_s+6)]

    for i in range(6+cl_len, 6+cl_len+e1_len):
        graph_dict[i] = [0, 0, 1, 0, 0, 0] + [0 for j in range(6, i - 1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, max_s+6)]

    for i in range(6+cl_len+e1_len, 6+cl_len+e1_len+cm_len):
        graph_dict[i] = [0, 0, 0, 1, 0, 0] + [0 for j in range(6, i - 1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, max_s+6)]

    for i in range(6+cl_len+e1_len+cm_len, 6+cl_len+e1_len+cm_len+e2_len):
        graph_dict[i] = [0, 0, 0, 0, 1, 0] + [0 for j in range(6, i - 1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, max_s+6)]

    for i in range(6+cl_len+e1_len+cm_len+e2_len, max_s+6-1):
        graph_dict[i] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, i - 1)] + \
                        [1, 1, 1] + [0 for j in range(i + 2, max_s+6)]

    graph_dict[max_s+6-1] = [0, 0, 0, 0, 0, 1] + [0 for j in range(6, max_s+6-2)] + [1, 1]

    return graph_dict


if __name__ == '__main__':

    trainfile = '/Users/shengbinjia/Documents/GitHub/GNN4ZeroShotRE/data/WikiReading/WikiReading_data.random.train.txt'
    testfile = '/Users/shengbinjia/Documents/GitHub/GNN4ZeroShotRE/data/WikiReading/WikiReading_data.random.test.txt'
    # GetGraph()

    reg = GetGraph_gat(max_s=13, pad_left=2, e1_l=1, e1_r=3, e2_l=5, e2_r=8)
    for i in reg.keys():
        print(i)
        print(reg[i])
