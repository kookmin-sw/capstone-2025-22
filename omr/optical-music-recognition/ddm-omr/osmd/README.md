# how to use

## 1. 경로 설정

`.env`에서 `DATASET_PATH`에 사용할 데이터셋 경로 지정

(예시)

```
DATASET_PATH = "./osmd-dataset"
```

- `DATASET_PATH`의 구조는 아래를 따라야 함
  ```
  ㄴ dataset-path
      ㄴ folder1
          ㄴ XXXX.xml
      ㄴ folder2
          ㄴ XXXX.xml
      ㄴ folder3
          ㄴ XXXX.xml
      ...
  ```
- 해당 경로에 이미지 파일과 커서 위치 등을 담은 파일이 저장되므로 필요하다면 기존 파일은 사본을 저장해 둘 것.
- 작업 완료 후 폴더 내용은 아래와 같음
  ```
  ㄴ dataset-path
      ㄴ folder1
          ㄴ XXXX.xml
          ㄴ XXXX.png
          ㄴ XXXX.json
      ㄴ folder2
          ㄴ XXXX.xml
          ㄴ XXXX.png
          ㄴ XXXX.json
      ㄴ folder3
          ㄴ XXXX.xml
          ㄴ XXXX.png
          ㄴ XXXX.json
      ...
  ```

## 2. 빌드

실행 위치: `./osmd`

```
sudo docker compose build
```

## 3. 실행

실행 위치: `./osmd`

```
sudo docker compose up
```
