---
title: VS code Anaconda 연동
author: JaewukLee-S
date: 2022-05-04 16:30:00 +0900
categories: [Anaconda]
tags: [install, win10, anaconda, python]
---

# 1. 확장 설치

![vscode extensions](/assets/img/post/tech/2022/05/vscode-anaconda/vscode-extensions.png)
- VS code 실행 후 Extenstions 탭 클릭
- Python 과 Code Runner 확장 설치
- VS code 재실행

# 2. 연동

- ctrl + shift + p 누른 후 Python: Select Interpreter 선택
- Python 환경 선택
- ctrl + ` 눌러서 terminal 탭을 실행하면 자동으로 명령어를 입력하고 환경에 접속한 상태 확인 가능

# 3. 에러 발생 시

> ommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
- [관련](https://kaiercorp.github.io/posts/conda-activate/)