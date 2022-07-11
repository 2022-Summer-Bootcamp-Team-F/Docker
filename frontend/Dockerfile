# docker hub에서 image를 pull
FROM node:16.13.0-alpine

# set working directory
# 작업을 수행할 디렉토리를 지정
WORKDIR /frontend

# 현재 경로에 있는 것을 frontend로 복사
COPY . /frontend

# 이미지가 빌드 될때 실행
RUN npm install
