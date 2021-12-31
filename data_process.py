import jieba
import pandas as pd
import numpy as np



# 1. pandas读入数据
import torch
from gensim.models import KeyedVectors

data = pd.read_table("./data/train_new.txt", header=None)
# print(data.head())
data = data.dropna()    # 取出空行
# 取两列句子和标签
sentences1_list = data[1].values.tolist()   # sentence1
sentences2_list = data[2].values.tolist()   # sentence2
label_list = data[3].values.tolist()        # label

# 2. 取出第一二个句子并转为向量
def seg_words_to_vec(sentences_list):
    """
    分词并将转为向量
    :param sentences_list:
    :return: vec_arr:array
    """
    # 分词后的外层空列表
    sentences_words_list = []
    # 分词
    for i in range(len(sentences_list)):
        sentences_words_list.append(list(jieba.lcut(sentences_list[i], cut_all=False)))
    # 分词后，词数最多的句子的词数
    max_sentences_words_list_len = 0    # 44/36
    for words_list in sentences_words_list:
        if len(words_list) > max_sentences_words_list_len:
            max_sentences_words_list_len = len(words_list)
    # sentences_words_list -> vec
    wv = KeyedVectors.load_word2vec_format("sgns.renmin.bigram-char", binary=False)
    vec_list = []   # 列表形式存放向量
    for words_list in sentences_words_list:
        num = 0     # 记录列表中词的数量
        vec_list_temp = []  # 词转换为向量后的临时存放列表
        for word in words_list:
            try:
                vec_list_temp.append(wv[word])
            except:
                vec_list_temp.append(np.random.uniform(-0.05, 0.05, 300))  # 出错的词用生成随机向量补齐
            finally:
                num += 1
        while num < max_sentences_words_list_len:
            vec_list_temp.append(np.zeros(300))  # 长度不足的用0补齐
            num += 1
        vec_list.append(vec_list_temp)
        vec_arr = np.array(vec_list, dtype=np.float32)
    return vec_arr

#
sentences1_tensor = torch.from_numpy(seg_words_to_vec(sentences1_list))
sentences2_tensor = torch.from_numpy(seg_words_to_vec(sentences2_list))
torch.save(sentences1_tensor, "sentences1_new.pt")
torch.save(sentences2_tensor, "sentences2_new.pt")
label_vec = np.array(label_list)
torch.save(label_vec, "labels_new.pt")

