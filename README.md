## 1. Chirpy Theme 설정

주요 설정 목록들입니다. 일부 값은 변경이 필요합니다.

1. _config.yml
- baseurl
  - 블로그 내에서 상대경로 사용 시 등록. 기본값 빈 값으로 둡시다
- timezone: Asia/Seoul
- title
  - 블로그 타이틀
- tagline: Kaier Tech Blog
  - 블로그 서브 타이틀
- github
  - 깃헙 아이디
- twitter
- social
  - 각종 소셜 아이디 설정
- google_site_verification
- google_analytics
  - 구글 연동 필요시 설정합시다.
- theme_mode
  - dark/light 중 선택 가능하며, 기본값으로 두면 시스템 설정을 따릅니다.
- toc: true
  - 포스트 내용을 제목 태그에 따라 목록화 해주는 기능입니다.
- paginate: 10
  - 목록에서 한 페이지에 보여주는 포스트 갯수입니다.
- img_cdn: ''
  - 이미지 파일을 외부 사이트에 등록한 경우 사용합니다.

2. .github/
- workflows 폴더를 제외하고 모두 삭제합니다.
- workflows 폴더에서 commitlint.yml, pages-deploy.yml.hook 파일을 제외하고 모구 삭제합니다.
- pages-deploy.yml.hook -> pages-deploy.yml로 변경합니다.

3. pages-deploy.yml
- branchs: develop
- 배포 브랜치를 지정합니다. 우리는 develop이 메인 브렌치이므로 develop으로 변경해줍니다.

## 2. Author 등록
작성자 정보를 미리 등록해두고, 포스트 작성 시 선택해 줍니다.
- _data/authors.yml
- 필요시 추가합니다.

## 3. 포스팅
- _posts 폴더에 포스트 파일을 추가합니다.
- 관리상 편의를 위해 폴더 구조를 사용할 필요가 있습니다. (브레인스토밍 필요)
- 파일명은 yyyy-MM-dd-제-목-띄어쓰기-없이.md
- 이미지 파일 등록이 필요한 경우
  - assets/img/post 에 폴더 구조로 저장한 뒤 링크합니다.
- [포스트 예시](https://kaiercorp.github.io/posts/vscode-anaconda/)
  - title : 포스트 제목
  - author : authors.yml에 등록된 아이디를 입력합니다.
  - date : 양심껏 날짜를 입력합니다.
  - categories: [main, sub] 형태로 최대 2depth 카테고리징이 됩니다.
  - tags: [a, b, c] 형태로 다양한 태그를 달 수 있습니다.
  - 그 외에 Markdown 형식으로 포스팅 하면 됩니다.

## 4. Social
_data/contact.yml
- Social ID 등록 가능
- 현재 github, email, rss 만 등록함
- twitter, stackoverflow linkedin 추가 가능

## TODO
- Jekyll 환경 설정
- Jekyll Theme 적용
- Github 배포