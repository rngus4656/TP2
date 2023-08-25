# TeamProject2 - Reccomendation

참고한 모델

GNN : https://github.com/SeongBeomLEE/RecsysTutorial/blob/main/LightGCN/LightGCN.ipynb
SASRec : https://github.com/pmixer/SASRec.pytorch.git

# GNN

코랩에서 사용하려면 config, args에서 데이터셋 경로, 모델경로 등만 다시 설정해주면됩니다

로컬에서 사용시 git clone하고 pip install -r requirements.txt 후
주피터노트북 켜서 전체실행 누르시면 실행됩니다

# SASRec
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --lr 0.001 --num_epochs 200
실행시 출력값이 iter값마다 나올경우 너무 많은 출력이 나오게되어 epoch마다 평균loss값이 나오게 수정하였고
나머지 코드는 기존 코드와 동일합니

파이썬 버전 3.10
