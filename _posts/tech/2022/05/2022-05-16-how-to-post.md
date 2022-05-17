---
title: 블로그 포스팅 방법
author: JaewukLee-S
date: 2022-05-16 16:30:00 +0900
categories: [Git]
tags: [tips, git, blog]
---

## [1] 포스팅 포맷

### 1. 최상단 포스트 정보

- title: 포스트 제목
- author: 작성자 ID
  - _data/authors.yml에 등록된 사용자 정보가 매핑됩니다.
  - 아이디는 일관성을 위해  git/slack ID로 임의로 등록해두었습니다.
  - 그 외에 트위터 아이디나 개인 홈페이지를 연동 할 수 있습니다.
- date: 작성일자를 yyyy-MM-dd HH:mm:ss 형태로 양심껏 입력합니다.
- categories: [MAIN, SUB] 형태로 최대 2depth 카테고리가 가능합니다. 현재 포스트는 1depth만 갖고 있습니다.
- tags: [a, b, c, ..] 형태로 다수의 태그를 설정 할 수 있습니다. 현재 포스트는 tips, git, blog 3개의 태그가 설정되었습니다.


### 2. 포스트 내용

- Markdown 형태로 포스트를 작성하면 됩니다.
- 파일
  - _posts/tech/yyyy/MM/ 위치에 'yyyy-MM-dd-제목-띄어쓰기-없이.md 생성
  - tech가 아닌 다른 구분이 필요한 경우 추가할 수 있습니다. ex) playday
- 이미지 업로드
  - assets/img/post 위치에 포스트 디렉토리 구조와 같게 이미지 파일들을 위치시킵니다.
  - ```![이미지이름](/assets/img/post/이미지 path/이미지파일명)``` 형태로 포스트에 포함시킵니다.
![로고](/assets/img/kaier.png)

### 3. md 파일 내용
현재까지 이 포스트의 내용은 다음과 같습니다.

```
---
title: 블로그 포스팅 방법
author: JaewukLee-S
date: 2022-05-15 13:30:00 +0900
categories: [Git]
tags: [tips, git, blog]
---

## [1] 포스팅 포맷

### 1. 최상단 포스트 정보

- title: 포스트 제목
- author: 작성자 ID
  - _data/authors.yml에 등록된 사용자 정보가 매핑됩니다.
  - 아이디는 일관성을 위해  git/slack ID로 임의로 등록해두었습니다.
  - 그 외에 트위터 아이디나 개인 홈페이지를 연동 할 수 있습니다.
- date: 작성일자를 yyyy-MM-dd HH:mm:ss 형태로 양심껏 입력합니다.
- categories: [MAIN, SUB] 형태로 최대 2depth 카테고리가 가능합니다. 현재 포스트는 1depth만 갖고 있습니다.
- tags: [a, b, c, ..] 형태로 다수의 태그를 설정 할 수 있습니다. 현재 포스트는 tips, git, blog 3개의 태그가 설정되었습니다.


### 2. 포스트 내용

- Markdown 형태로 포스트를 작성하면 됩니다.
- 파일
  - _posts/tech/yyyy/MM/ 위치에 'yyyy-MM-dd-제목-띄어쓰기-없이.md 생성
  - tech가 아닌 다른 구분이 필요한 경우 추가할 수 있습니다. ex) playday
- 이미지 업로드
  - assets/img/post 위치에 포스트 디렉토리 구조와 같게 이미지 파일들을 위치시킵니다.
  - ```![이미지이름](/assets/img/post/이미지 path/이미지파일명)``` 형태로 포스트에 포함시킵니다.
![로고](/assets/img/kaier.png)

### 3. md 파일 내용
```

## 2. github.com에서 포스팅하기

- [github](https://github.com/kaiercorp/kaiercorp.github.io) 접속
- _posts/ 로 이동
- Add files - Create new file로 포스트 작성
![이미지](/assets/img/post/tech/2022/05/hwo-to-post/how-to-post-01.png)

- yyyy-MM-dd-제목.md 형태로 파일명 작성
- 포스트 내용 작성
![이미지1](/assets/img/post/tech/2022/05/hwo-to-post/how-to-post-02.png)

- 포스트 커밋 : 커밋 메시지 입력 후 저장
![이미지2](/assets/img/post/tech/2022/05/hwo-to-post/how-to-post-03.png)


## 3. 로컬에서 포스팅하기
**IDE에서 md 미리보기를 지원하는 경우, 굳이 jekyll을 설치하지 않아도 됨**

1. Ruby 설치
  - rubyinstaller.org/downloads/
2. jekyll 설치
  ```
  gem install jekyll
  ```
3. bundler 설치
  ```
  gem install bundler
  ```
4. clone github
  ```
  git clone https://github.com/kaiercorp/kaiercorp.github.io.git
  ```
5. 실행
  ```
  cd kaiercorp.github.io
  bundle exec jekyll serve
  ```
- localhost:4000/
![이미지2](/assets/img/post/tech/2022/05/hwo-to-post/how-to-post-04.png)

6. 포스팅
  - _posts/ 위치에 포스팅 파일 작성 후, localhost:4000 에서 확인 한 뒤 커밋

