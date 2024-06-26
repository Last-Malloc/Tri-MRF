# 在2D-TAN提供的Charades-STA数据集上进行处理


import csv
import json
import os.path
import pickle


def read_text(path, post_pro_fun=None):
    result = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            if len(line) == 0:
                continue
            if post_pro_fun:
                line = post_pro_fun(line)
            result.append(line)

    return result


def create_charades_sta_len(show_content=False):
    if os.path.exists("charades_sta_len.pkl"):
        if show_content:
            with open("charades_sta_len.pkl", "rb") as f:
                print(pickle.load(f))
        print("charades_sta_len.pkl 已存在")
    else:
        durations = {}
        with open("Charades_v1_train.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                durations[row['id']] = float(row['length'])
        with open("Charades_v1_test.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                durations[row['id']] = float(row['length'])
        with open("charades_sta_len.pkl", "wb") as f:
            pickle.dump(durations, f)
        if show_content:
            print(durations)
        print("charades_sta_len.pkl 创建成功")


def create_anno_pairs(split, show_content=False):
    if os.path.exists("charades_sta_anno_pairs_{}.pkl".format(split)):
        if show_content:
            with open("charades_sta_anno_pairs_{}.pkl".format(split), "rb") as f:
                print(pickle.load(f))
        print("charades_sta_anno_pairs_{}.pkl 已存在".format(split))
    else:
        with open("charades_sta_len.pkl", "rb") as f:
            durations = pickle.load(f)
        anno_pairs = []
        lines = read_text("charades_sta_{}.txt".format(split))
        for line in lines:
            front, back = line.split('##')
            vid, start, end = front.split(' ')
            start = float(start)
            end = float(end)
            sentence = back.replace('.', '').lower()
            duration = durations[vid]
            anno_pairs.append(
                {
                    'video': vid,
                    'duration': duration,
                    'times': [start, end],
                    'description': sentence,
                }
            )
        with open("charades_sta_anno_pairs_{}.pkl".format(split), "wb") as f:
            pickle.dump(anno_pairs, f)
        if show_content:
            print(anno_pairs)
        print("charades_sta_anno_pairs_{}.pkl 创建成功".format(split))


def create_words_vocab_charades_sta():
    if os.path.exists('words_vocab_charades_sta.json'):
        with open('words_vocab_charades_sta.json', 'r') as f:
            print('字符集长度：{}'.format(len(json.load(f)["words"])))
        print('words_vocab_charades_sta.json 已存在')
        return
    js = set()
    with open("charades_sta_anno_pairs_train.pkl", "rb") as f:
        anno_pairs = pickle.load(f)
        for item in anno_pairs:
            description = item['description']
            words = description.split()
            for word in words:
                js.add(word)
    with open("charades_sta_anno_pairs_test.pkl", "rb") as f:
        anno_pairs = pickle.load(f)
        for item in anno_pairs:
            description = item['description']
            words = description.split()
            for word in words:
                js.add(word)

    js = list(js)
    js.insert(0, "PAD")
    js.append("UNK")
    with open('words_vocab_charades_sta.json', 'w') as f:
        json.dump({"words": js}, f)
    print('words_vocab_charades_sta.json 创建成功')


if __name__ == '__main__':
    # create_charades_sta_len()

    create_anno_pairs('train')
    create_anno_pairs('test')
    create_words_vocab_charades_sta()


