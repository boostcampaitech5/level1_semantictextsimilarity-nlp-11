# 🏆 Level 1 Projects :: STS(Semantic Text Similarity)
>의미 유사도 판별(Semantic Text Similarity, STS)이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 태스크입니다.

## 🧑🏻‍💻 Team Introduction & Members 

> 네이버까지 들어가조 [ NLP 11조 ] 

### Members
강민재|김주원|김태민|신혁준|윤상원|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/39152134?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/81630351?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96530685?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/96534680?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/38793142?v=4' height=80 width=80px></img>
[Github](https://github.com/mjk0618)|[Github](https://github.com/Kim-Ju-won)|[Github](https://github.com/taemin6697)|[Github](https://github.com/jun048098)|[Github](https://github.com/SangwonYoon)
kminjae618@gmail.com|kjwt1124@hufs.ac.kr|taemin6697@gmail.com|jun048098@gmail.com|iandr0805@gmail.com

## 💻 Baseline Code 사용법
```bash
#학습 명령어 
python3 code/train.py
#예측 명령어 output.csv 생성
python3 code/inference.py 
```

## 📐 Github Push Ground Rule

1. git add 이전에 repository에 올라가면 안되는 파일들(데이터)이 `.gitignore`안에 들어있는지 확인하기
    -  만약 `.gitignore`에 없으면 파일을 추가해주세요. 
    - 기본적으로 data안에 들어가 있는 모든 파일들은 push 했을 때 remote 주소로 올라가지 않으니 train, test, dev 파일들은 생성시 data폴더를 만들어 해당 폴더 내부에 넣어주시는 걸 추천합니다.

2. `git commit` 이전에 본인의 branch가 맞는지 확인해주세요. (branch가 본인의 initial과 같은지 확인) 만약 아니라면 아래 명령어를 통해 본인의 브랜치로 반드시 변경해주세요. 아래는 초기 설정 예시입니다.
```bash
# git checkout [본인브랜치 이름(이니셜)]
# 예시 
git switch -c KJW
```
- 이후 push할 origin 브랜치를 연결해줍니다. 
```bash
# git push --set-upstream [본인브랜치 이름(이니셜)]
git push --set-upstream origin KJW
```

- 중간에 다른 브랜치로 바꿔서 코드를 확인하고 싶으시면 아래 명령어를 이용하시면 됩니다. 브랜치를 옮겨서 확인하시고, 자기 branch로 switch해서 돌아오는걸 잊지 마세요. 
```bash
# git checkout [본인브랜치 이름(이니셜)]
# 예시 
git switch KJW
```


3. 이번 프로젝트의 commit 단위는 **한번 submission**할 때 마다  커밋 하는 것을 원칙으로 합니다. 아래 예시를 보고 **코드 수정 내용, 점수, 제출 시간**이 들어가도록 commit 해주시면 됩니다. 

```bash
# git commit -am [코드 수정 사항] [public score 점수] [시간(연도.월.일 시간:분)]
# 예시 : 
git commit -am "Use HuggingFace Alectrica model score 97 23.04.11 18:50"
```

## 📁 프로젝트 구조
```
level1_semantictextsimilarity-nlp-11
 ┣ code
 ┣ data
 ┃ ┣ dev.csv
 ┃ ┣ test.csv
 ┃ ┗ train.csv
 ┣ .gitignore
 ┗ Readme.md
 ```
