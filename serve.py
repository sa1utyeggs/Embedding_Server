# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install fastapi uvicorn
"""
import argparse
import os
import sys
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.middleware.cors import CORSMiddleware
from text2vec import Similarity

sys.path.append('..')

pwd_path = os.path.abspath(os.path.dirname(__file__))
use_cuda = torch.cuda.is_available()
logger.info(f'use_cuda:{use_cuda}')
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="./model/text2vec-base-chinese_ATEC_128DIM",
                    help="Model save dir or model name")
args = parser.parse_args()
vector_model = SentenceTransformer(args.model_name_or_path)
sim_model = Similarity(args.model_name_or_path)
# s_model = SentenceModel(args.model_name_or_path)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


class EmbeddingRequest(BaseModel):
    sentences: List[str] = []


class SimilarityRequest(BaseModel):
    sentences1: List[str] = []
    sentences2: List[str] = []


@app.post('/embedding')
async def embedding(body: EmbeddingRequest):
    sentences = body.sentences
    # https://www.sbert.net/docs/package_reference/SentenceTransformer.html?highlight=stop
    # batch_size – the batch size used for the computation
    sentence_embeddings = vector_model.encode(sentences, batch_size=8, show_progress_bar=True,
                                              normalize_embeddings=True)
    # print(type(sentence_embeddings), sentence_embeddings.shape)

    # json
    results = {"rows": []}
    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        # print("Sentence:", sentence)
        # print("Embedding shape:", embedding.shape)
        # print("Embedding head:", embedding[0:5])

        # json
        row = {'vector': embedding.tolist()}
        results['rows'].append(row)
        # print()

    print(results)
    return results


@app.post('/similarity')
async def similarity(body: SimilarityRequest):
    scores = sim_model.get_scores(body.sentences1, body.sentences2)
    # json
    scores.tolist()
    results = {"scores": scores.tolist()}
    return results


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
