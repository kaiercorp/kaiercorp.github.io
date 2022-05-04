---
title: Win 10 conda activate Error
author: Jaewuk Lee
date: 2022-05-04 16:11:00 +0900
categories: [Anaconda]
tags: [tips, error, vscode, win10]
---

## 문제 상황

Anaconda 설치 및 가상 환경 생성 후 VS Code에서 가상환경 연동 시 에러 발생함.

&gt; activate torch_book

> ommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.

- 환경변수 설정 되었음에도 에러 발생함

## 원인

powershell의 정책 설정이 안되어서 발생하는 에러

## 처리

PowerShell을 관리자 모드로 실행 후 다음 명령어 실행
> set-ExecutionPolicy RemoteSigned