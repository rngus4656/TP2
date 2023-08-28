# python test.py --user_id 1 --dataset=ratings_enc_ts --train_dir=default --maxlen 200 --state_dict_path=./ratings_enc_ts_default/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth

import torch
import numpy as np
from model import SASRec
from utils import *
from main import *
import argparse
import pandas as pd


def get_user_seq(dataset, user_id):
    user_train, _, _, _, _ = dataset
    return user_train.get(user_id, [])

def recommend(model, user_seq, item_num, device='cpu'):
    user_seq = np.array(user_seq)
    user_seq = np.expand_dims(user_seq, 0)  # (1, seq_len)

    # 모든 영화에 대한 점수 예측
    item_indices = list(range(1, item_num + 1))
    scores = model.predict([0], user_seq, item_indices).squeeze(0)
    
    # 점수에 따라 영화 정렬
    recommended_items = scores.argsort(descending=True)[:10].cpu().numpy()

    return recommended_items

def load_movie_titles(dat_path="./data/movies.dat"):
    movie_id_to_title = {}
    with open(dat_path, 'r', encoding='ISO-8859-1') as f:  # 인코딩은 파일에 따라 다를 수 있습니다.
        for line in f.readlines():
            parts = line.split('::')  # ::로 구분된 파일이라고 가정합니다.
            movie_id = int(parts[0])
            title = parts[1]
            movie_id_to_title[movie_id] = title
    return movie_id_to_title

def load_encoders(file_path):
    """인코딩 했던 정보를 다시 가져오기 위한 함수"""
    df = pd.read_csv(file_path, sep=' ', header=None, names=['userId', 'movieId'])
    user_encoder = {original: idx for idx, original in enumerate(df['userId'].unique(), start=1)}
    movie_encoder = {original: idx for idx, original in enumerate(df['movieId'].unique(), start=1)}

    # decoder는 encoder의 반대입니다.
    user_decoder = {idx: original for original, idx in user_encoder.items()}
    item_decoder = {idx: original for original, idx in movie_encoder.items()}

    return user_decoder, item_decoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--user_id', default=None, type=int)  # user_id 인자 추가
    
    args = parser.parse_args()

    # 설정 및 모델 로드
    dataset = data_partition(args.dataset)
    _, _, _, usernum, itemnum = dataset

    model_args = argparse.Namespace()
    model_args.hidden_units = args.hidden_units
    model_args.maxlen = args.maxlen
    model_args.dropout_rate = args.dropout_rate
    model_args.num_blocks = args.num_blocks
    model_args.num_heads = args.num_heads
    model_args.device = args.device

    model = SASRec(usernum, itemnum, model_args)
    model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()

    user_seq = get_user_seq(dataset, args.user_id)  # user_id에 해당하는 상호작용을 가져옵니다
    recommended_items = recommend(model, user_seq, itemnum)
    print("Recommended items:", recommended_items)

    movie_id_to_title = load_movie_titles()

    user_decoder, item_decoder = load_encoders('./data/ratings_enc_ts.txt')

    # 사용자가 이미 본 영화를 추천에서 제외하기
    already_watched = set(user_seq)
    
    # 모든 영화 중에서 사용자가 이미 본 영화를 제외하고 상위 10개를 가져오는 로직
    user_seq_array = np.array(user_seq)
    user_seq_array = np.expand_dims(user_seq_array, 0)  # (1, seq_len)
    scores = model.predict([0], user_seq_array, list(range(1, itemnum + 1))).squeeze(0)
    recommended_items = [x for x in scores.argsort(descending=True).cpu().numpy() if x not in already_watched][:10]

    print("Movie ID:", recommended_items)

    movie_id_to_title = load_movie_titles()

    for item in recommended_items:
        movie_id = item_decoder.get(item, "Unknown")  # item_decoder로 원래 movie_id를 가져옵니다.
        title = movie_id_to_title.get(movie_id, "Unknown")
        print(f"Movie Title: {movie_id}, Title: {title}")

