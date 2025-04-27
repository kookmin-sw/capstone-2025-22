#!/bin/bash

# 소스 폴더와 대상 폴더를 지정합니다.
source_folder="./before"
destination_folder="./after"

# 대상 폴더가 없는 경우 생성합니다.
mkdir -p "$destination_folder"

# 소스 폴더의 모든 파일을 대상 폴더로 복사합니다.
cp -r "$source_folder"/* "$destination_folder"

sed -i 's/pitch/unpitched/g' "$destination_folder"/*
# sed -i 's/<step>/<display-step>/g' "$destination_folder"/*
sed -i 's/step>/display-step>/g' "$destination_folder"/*
# sed -i 's/<octave>/<display-octave>/g' "$destination_folder"/*
sed -i 's/octave>/display-octave>/g' "$destination_folder"/*

rm -r "$source_folder"/*

echo "변경이 완료되었습니다."


