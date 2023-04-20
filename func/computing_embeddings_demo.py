# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""
import hashlib
import json
import sys

import sklearn
import torch

sys.path.append('..')
from text2vec import SentenceModel, EncoderType
from text2vec import Word2Vec
from sentence_transformers import models, SentenceTransformer


def compute_emb(model, sentences):
    # Embed a list of sentences

    sentence_embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)

    # normalize
    # sentence_embeddings = sklearn.preprocessing.normalize(sentence_embeddings, X=[-1,1])

    # json
    map = {"rows": []}
    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        # print("Sentence:", sentence)
        # print("Embedding shape:", embedding.shape)
        # print("Embedding head:", embedding[0:5])

        # Tips
        # 此处必须声明 encode
        # 若写法为hl.update(str)  报错为： Unicode-objects must be encoded before hashing

        # json
        row = {'text': sentence, 'vector': embedding.tolist()}
        map['rows'].append(row)
        print()
    return map
    # with open("./sample.json", "w") as f:
    #     json.dump(map, f, ensure_ascii=False)
    #     print("加载入文件完成...")


if __name__ == "__main__":
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    # t2v_model = SentenceModel("shibing624/text2vec-base-chinese_128DIM",
    #                           max_seq_length=128,
    #                           encoder_type=EncoderType.FIRST_LAST_AVG)
    sentences = [
        '登陆失败如何解决',
        '用户登陆失败怎么办',
        '登陆失败时可以通过联系客服、申诉等方式进行处理',
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
        'This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.'
    ]
    t2v_model = SentenceTransformer("shibing624/text2vec-base-chinese_128DIM")
    compute_emb(t2v_model, )

    # # 支持多语言的句向量模型（Sentence-BERT），英文语义匹配任务推荐，支持fine-tune继续训练
    # sbert_model = SentenceModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #                             encoder_type=EncoderType.MEAN)
    # compute_emb(sbert_model)
    #
    # # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
    # w2v_model = Word2Vec("w2v-light-tencent-chinese")
    # compute_emb(w2v_model)
