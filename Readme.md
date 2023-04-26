# 🏆 Level 1 Project :: STS(Semantic Text Similarity)

### 📜 Abstract
> 부스트 캠프 AI-Tech 5기 NLP Level 1 기초 프로젝트 경진대회로, Dacon과 Kaggle과 유사항 대회형 방식으로 진행되었습니다. 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 N21 자연어처리 Task인 의미 유사도 판별(Semantic Text Similarity, STS)를 주제로, 모든 팀원이 데이터 전처리부터 모델의 하이퍼파라미터 튜닝에 이르기까지 AI 모델링의 전과정을 모두가 End-to-End로 협업하는 것을 목표로 프로젝트를 진행했습니다. 

<br>

## 🎖️Project Leader Board 
![public_1st](https://img.shields.io/static/v1?label=Public%20LB&message=1st&color=yellow&logo=naver&logoColor=white") ![private_2nd](https://img.shields.io/static/v1?label=Private%20LB&message=2nd&color=silver&logo=naver&logoColor=white">)
- 🥇 Public Leader Board
<img width="1089" alt="public_leader_board" src="https://user-images.githubusercontent.com/81630351/234538736-4a1f4447-2aed-4187-9ac9-cf03cc65c507.png">
- 🥈Private Leader Board 
<img width="1089" alt="private_leader_board" src="https://user-images.githubusercontent.com/81630351/234539421-144d8ea6-a8ad-47c8-bee4-45c488ca4cfc.png">

- [📈 NLP 11조 Project Wrap-Up report 살펴보기](https://github.com/boostcampaitech5/level1_semantictextsimilarity-nlp-11/files/11331465/NLP.11.Wrap-Up._.pdf)

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team name : 윤슬 [ NLP 11조 ] 

### 👨🏼‍💻 Members
강민재|김주원|김태민|신혁준|윤상원|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/39152134?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/81630351?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96530685?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96534680?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/38793142?v=4' height=80 width=80px></img>
[Github](https://github.com/mjk0618)|[Github](https://github.com/Kim-Ju-won)|[Github](https://github.com/taemin6697)|[Github](https://github.com/jun048098)|[Github](https://github.com/SangwonYoon)
kminjae618@gmail.com|kjwt1124@hufs.ac.kr|taemin6697@gmail.com|jun048098@gmail.com|iandr0805@gmail.com

### 🧑🏻‍🔧 Members' Role
> 대부분의 팀원들이 첫 NLP 도메인의 프로젝트인만큼 명확한 기준을 가지고 업무를 구분한 것보다 다양한 인사이트를 기르기 위해 데이터 전처리부터 모델 튜닝까지 End-to-End로 경험하는 것을 목표로 하여 협업을 진행했습니다. 따라서 각자 튜닝할 모델을 할당하여 하이퍼 파라미터 튜닝을 하고 데이터 전처리, 증강 등 본인의 아이디어를 구현하되 서로의 내용이 겹치지 않도록 분업을 하여 프로젝트를 진행했습니다.

| 이름 | 역할 |
| --- | --- |
| **`강민재`** | **모델 튜닝**(`electra-kor-base , koelectra-base-v3-discriminator`),**데이터 증강**(`back translation / switching sentence pair /임의글자삽입및제거`),**데이터 전처리 실험**(`레이블 정수화 및 노이즈추가`),**Ensemble 실험**(`output 평균, 표준편차활용`),**EDA**(`글자수 기반 데이터 분포 분석`) |
| **`김태민`** | **Hugging Face 기반 Baseline 코드 작성** , **Task에 적합한 모델 Search 및 분배** , **모델 실험 총괄** , **데이터 전처리 실험**(`Random Token Masking , Label Random Noise, Fill Random Token Mask, Source Tagging`), **Custom Loss 실험**(`Binary Cross Entropy + Focal Loss`),**모델 튜닝**(`xlm-roberta-large, electra-kor-base`),**모델 Ensemble** |
| **`김주원`** | **모델 튜닝**(`kobigbird-bert-base, electra-kor-base`),**EDA**(`라벨 분포 데이터분석`),**EDA 기반 데이터 증강 아이디어 제시** , **데이터 증강**(`Easy Augmented DataSR 증강`),**팀 협업 프로세스 관리**(`Github 팀관리+ 팀 Notion 페이지관리`) ,**Custom Loss 실험**(`RMSE`) |
| **`윤상원`** | **모델 튜닝**(`koelectra-base-finetuned-nsmc, KR-ELECTRA-discriminator 모델튜닝`),**데이터 증강**(`label rescaling, 단순복제데이터증강, 어순도치데이터증강, under sampling + swap sentence + copied sentence + uniform distribution + random noise`),**모델 Ensemble** |
| **`신혁준`** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **모델 튜닝**(`KR-ELECTRA-discriminator, mdeberta-v3-base-kor-further`)**데이터 증강**(`맞춤법교정증강,EDA(Easy Data Augmentation) SR(Synonym Replacement)품사선택(명사, 조사) 교체+ swap sentence + copied sentence, Data Distribution`),**데이터 전처리 실험**(`맞춤법교정`) |

<br>

## 🖥️ Project Introduction 


|**프로젝트 주제**| **`Semantic Text Similarity (STS)` :** 두 텍스트가 얼마나 유사한지 판단하는 NLP Task|
| --- | --- |
|**프로젝트 구현내용**| 1. Hugging Face의 Pretrained 모델과STS 데이터셋을 활용해 두 문장의 0과 5사이의 유사도를 측정하는 AI모델을 구축 <br>2. 리더보드 평가지표인 피어슨 상관 계수(Pearson Correlation Coefficient ,PCC)에서 높은 점수(1에 가까운 점수)에 도달할 수 있도록 데이터 전처리, 증강, 하이퍼 파라미터 튜닝을 진행|
|**개발 환경**|**• `GPU` :** Tesla V100 서버 5개 (RAM32G) / K80, T4, and P100 랜덤 할당(RAM52G) /GeForce RTX 4090 로컬 (RAM 24GB), Rtx3060ti 8gb 로컬 2대 (RAM 8 GB)<br>**• `개발 Tool` :** PyCharm, Jupyter notebook, VS Code [서버 SSH연결], Colab Pro +, wandb |
|**협업 환경**|**• `Github Repository` :** Baseline 코드 공유 및 버전 관리, issue 페이지를 통하 실험 진행 <br>**• `Notion` :** STS 프로젝트 페이지를 통한 역할분담, 아이디어 브레인 스토밍, 대회관련 회의 내용 기록 <br>**• `SLACK, Zoom` :** 실시간 대면/비대면 회의|

## 📁 Project Structure
```
📦level1_semantictextsimilarity-nlp-11
 ┣ .gitignore
 ┣ config_yaml
 ┃ ┣ kykim.yaml
 ┃ ┣ snunlp.yaml
 ┃ ┣ test.yaml
 ┃ ┗ xlm_roberta_large.yaml
 ┣ data
 ┃ ┣ train.csv
 ┃ ┣ aug_train.csv
 ┃ ┣ dev.csv
 ┃ ┗ test.csv
 ┣ wordnet
 ┃ ┗ wordnet.pickle
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
>팀 협업을 위해 프로젝트 관련 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 팀 규칙을 정했으며, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획을 하여 진행했습니다.

**-a. `실험 관련 Ground Rule`** : 본인 실험 시작할 때, Github issue에 `"[score 점수(없다면 --)] 모델이름, data = 데이터 종류, 전처리 종류, 데이터 증강 종류"`양식으로 `issue`를 올린 뒤 실험을 시작한다.

**-b. `Commit 관련 Ground Rule`** : `git commit & push`는 한번 `실험할 때` 마다 진행한다. `코드 수정 내용, 점수, 관련된 issue`가 들어가도록 commit하고 개인 branch에 push한다.

**-c. `Submission 관련 Ground Rule:`** 각 사람별로 하루 submission 횟수는 `2회`씩 할당한다. 추가로 submission을 하고 싶으면 SLACK 단체 톡방에서 해당 날짜에 submission계획이 없는 혹은, 횟수가 남는 사람에게 물어봐서 여유가 된다면 추가 submission 가능하다.
<br>

## 🗓️ Project Procedure
- **`(1~3일차)`**: NLP 기초 대회 관련 대회 강의 및 스페셜 미션 완료 & 협업 관련 Ground Rule 설정
- **`(3~4일차)`**: STS Baseline 코드 완성 & EDA(Exploratory Data Analysis) & 전처리/증강 관련 아이디어 회의
- **`(5~14일차)`** : 전처리, 데이터 증강, 하이퍼 파라미터 튜닝 등 아이디어 구현 및 실험 진행

*아래는 저희 프로젝트 진행과정을 담은 Gantt차트 입니다. 

![road_map](https://user-images.githubusercontent.com/81630351/234539749-d2f7423d-9895-4aeb-a458-4ae593bc4db8.png)

<br>

## 💻 Getting Started

### ⚠️ Requirements
```bash
python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install transformers
pip3 install pandas
pip3 install scipy
pip3 install soynlp
pip3 install git+https://github.com/haven-jeon/PyKoSpacing.git
pip3 install git+https://github.com/jungin500/py-hanspell
pip3 install konlpy
```

### ⌨️ How To install Requirements
```bash
#필요 라이브러리 설치
pip install -r requirements.txt
```

### ⌨️How To Train 
```bash
# 데이터 증강
python3 augmentation.py
# train.py 코드 실행 : 모델 학습 진행
python3 train.py
```

### ⌨️ How To Infer output.csv
#예측 명령어 output.csv 생성
```bash
# infer.py 코드 실행 : 훈련된 모델 load + sample_submission을 이용한 train 진행
python3 infer.py 
```
