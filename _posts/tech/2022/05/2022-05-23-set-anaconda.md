---
title: 아나콘다 설치 및 설정 (TBU)
author: JaewukLee-S
date: 2022-05-23 15:30:00 +0900
categories: [Python, Anaconda]
tags: [tips, Python, anaconda]
---

## 1. 아나콘다 설치
- https://www.anaconda.com/products/individual 설치파일 다운로드
- 설치 후 환경 변수 설정
  - C:\Users\kaier\Anaconda3
  - C:\Users\kaier\Anaconda3\Library
  - C:\Users\kaier\Anaconda3\Scripts
- ![환경변수](/assets/img/post/tech/2022/05/pytorch-beginner/pytorch-beginner-02.png)

## 2. 가상 환경 생성
- cmd 또는 Anaconda Prompt 실행
- 가상환경 생성
```
> conda create -n torch_beginner python=3.9.0
```
-> torch_beginner 이라는 이름의 가상환경이 생성되며, 파이썬 버전은 3.9.0
- 가상환경 활성화
```
> conda env list
# conda environments:
#
base e:\Anaconda3
torch_beginner e:\Anaconda3\envs\torch_beginner

> conda activate torch_beginner
```

- 가상환경 삭제
```
> conda env remove -n torch_beginner
```

- 커널 설치 (가상환경 활성화 상태에서)
```
> pip install ipykernel
> python -m ipykernel install --user -0name torch_beginner
```

- 파이토치 설치
```
> conda install pytorch torchvision -c pytorch
```

- conda activate시 오류가 발생하면 https://kaiercorp.github.io/posts/conda-activate/
- vs code를 사용한다면 https://kaiercorp.github.io/posts/vscode-anaconda/