import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []   # 存储故事上下文（4句话拼接）
        ct1 = []  # 存储第一个候选结尾
        ct2 = []  # 存储第二个候选结尾
        y = []    # 存储标签（0 或 1）
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:  # 跳过表头
                s = ' '.join(line[1:5])  # 拼接4句话作为上下文
                c1 = line[5]             # 第一个候选结尾
                c2 = line[6]             # 第二个候选结尾
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)  # 标签转换为0和1，原来是1和2
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)
    # (trX1, trX2, trX3, trY)训练集：故事上下文、候选结尾1、候选结尾2、标签
    # (vaX1, vaX2, vaX3, vaY)验证集：故事上下文、候选结尾1、候选结尾2、标签
    # (teX1, teX2, teX3)测试集：故事上下文、候选结尾1、候选结尾2（无标签