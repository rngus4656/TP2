import pandas as pd

# ratings.dat 파일을 읽습니다.
col_names = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('./data/ratings.dat', sep='::', names=col_names, engine='python')

# 타임스탬프를 기준으로 데이터를 정렬합니다.
df = df.sort_values(by=['userId', 'timestamp'])
print(df)
# 사용자 ID와 영화 ID를 라벨 인코딩합니다.
user_encoder = {original: idx for idx, original in enumerate(df['userId'].unique(), start=1)}
movie_encoder = {original: idx for idx, original in enumerate(df['movieId'].unique(), start=1)}

df['userId'] = df['userId'].map(user_encoder)
df['movieId'] = df['movieId'].map(movie_encoder)

# .txt 파일로 저장합니다.
df[['userId', 'movieId']].to_csv('./data/ratings_enc_ts.txt', sep=' ', index=False, header=False)
df[['userId', 'movieId', 'rating', 'timestamp']].to_csv('./data/ratings_enc_ts_allcol.txt', sep=' ', index=False, header=False)
print(df)

