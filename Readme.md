# 🏆 Level 1 Project :: STS(Semantic Text Similarity)

### 📜 Abstract
> 부스트 캠프 AI-Tech 5기 NLP Level 1 기초 프로젝트 경진대회로, Dacon과 Kaggle과 유사항 방식으로 진행되었습니다. 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 N21 자연어처리 Task인 의미 유사도 판별(Semantic Text Similarity, STS)를 주제로 하여 진행하습니다. 모든 팀원이 데이터 전처리부터 모델의 하이퍼파라미터 튜닝에 이르기까지 AI 모델링의 전과정을 모두가 End-to-End로 협업하는 것을 목표로 프로젝트를 진행했습니다. 

## 🎖️Project Leader Board 
![public_1st](https://img.shields.io/static/v1?label=Public%20LB&message=1st&color=yellow&logo=naver&logoColor=white") ![private_2nd](https://img.shields.io/static/v1?label=Private%20LB&message=2nd&color=silver&logo=naver&logoColor=white">)
- 🥇 Public Leader Board
![Public Leader Board](./readme_img/public_leader_board.png)
- 🥈Private Leader Board 
![Private Leader Board](./readme_img/private_leader_board.png)

- [📈 NLP 11조 Project Wrap-Up report 살펴보기](https://github.com/boostcampaitech5/level1_semantictextsimilarity-nlp-11/blob/main/wrap-up_report/NLP%2011%EC%A1%B0%20Wrap-Up%20%EB%B3%B4%EA%B3%A0%EC%84%9C_%ED%8C%80%EC%B5%9C%EC%A2%85.pdf)

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 윤슬 [ NLP 11조 ] 

### Members
강민재|김주원|김태민|신혁준|윤상원|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/39152134?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/81630351?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96530685?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96534680?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/38793142?v=4' height=80 width=80px></img>
[Github](https://github.com/mjk0618)|[Github](https://github.com/Kim-Ju-won)|[Github](https://github.com/taemin6697)|[Github](https://github.com/jun048098)|[Github](https://github.com/SangwonYoon)
kminjae618@gmail.com|kjwt1124@hufs.ac.kr|taemin6697@gmail.com|jun048098@gmail.com|iandr0805@gmail.com

### Members' Role


## Project Introduction 

## 📁 Project Structure
```
📦level1_semantictextsimilarity-nlp-11
 ┣ .gitignore
 ┣ config_yaml
 ┃ ┣ kykim.yaml
 ┃ ┣ snunlpy.yaml
 ┃ ┣ test.yaml
 ┃ ┗ xlm_roberta_large.yaml
 ┣ data
 ┃ ┣ train.csv
 ┃ ┣ aug_train.csv
 ┃ ┣ dev.csv
 ┃ ┗ test.csv
 ┣ esnb
 ┃ ┗ esnb.csv
 ┣ output
 ┃ ┣ xlm_roberta_large.csv
 ┃ ┣ kykim.csv
 ┃ ┗ snunlp.csv
 ┣ .gitignore
 ┣ Readme.md
 ┣ augmentation.py
 ┣ dataloader.py
 ┣ esnb.py
 ┣ infer.py
 ┣ train.py
 ┗ utils.py
 ```

## 📐 Project Ground Rule



 ## 💻 Getting Started

 ### Requirements

 ### How to Train 

 ### How to Inference
```bash
#학습 명령어 
python3 code/train.py
#예측 명령어 output.csv 생성
python3 code/inference.py 
```
