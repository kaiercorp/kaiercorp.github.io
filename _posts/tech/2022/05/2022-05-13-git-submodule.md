---
title: Git Submodule 설정
author: JaewukLee-S
date: 2022-05-12 13:30:00 +0900
categories: [Git]
tags: [tips, git, submodule]
---

# 1. 서브 모듈 처음 추가

### - 서브 모듈 추가
```
~/workspace/parent_project
$ git submodule add https://github.com/gitname/child_project
```
- 부모 프로젝트의 디렉토리에서 git submodule 실행
- 자식 프로젝트의 repository명과 동일한 디렉토리가 생성됨

```
~/workspace/parent_project
git submodule add https://github.com/gitname/child_project sub_project
```
- repository명인 child_project 대신 sub_project 라는 디렉토리에 서브모듈이 등록됨


### - 서브 모듈 추가 결과
```
~/workspace/parent_project
$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    new file:   .gitmodules
    new file:   child_project
```
- .gitmodules : 서브모듈 목록 정보
- child_project : 서브모듈 디렉토리 (서브 모듈 repository 명으로 생성됨)

# 2. 서브 모듈이 있는 프로젝트 클론
1. 부모 프로젝트 클론
2. 서브 모듈 정보를 초기화 후 업데이트
```
~/workspace/parent_project
$ git sumbmodule init
$ git submodule update
```
