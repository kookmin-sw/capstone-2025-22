package com.capstone.service;

import com.capstone.dto.score.OnsetMatchResult;

import java.util.Arrays;
import java.util.List;

public class DTWMatcher {

    public static OnsetMatchResult match(List<Double> userOnset, List<Double> answerOnset, double threshold, double weight) {
        int n = userOnset.size();
        int m = answerOnset.size();
        double[][] dtwMatrix = new double[n + 1][m + 1];  // DTW 행렬 초기화

        answerOnset = answerOnset.stream().map(x -> x * weight).toList();

        // dtwMatrix의 첫 번째 열과 첫 번째 행을 무한대로 설정
        for (int i = 0; i <= n; i++) {
            dtwMatrix[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 0; j <= m; j++) {
            dtwMatrix[0][j] = Double.POSITIVE_INFINITY;
        }
        dtwMatrix[0][0] = 0;  // 시작점은 0으로 설정

        // DTW 매트릭스 채우기
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double cost = Math.abs(userOnset.get(i - 1) - answerOnset.get(j - 1));  // 비용 계산
                dtwMatrix[i][j] = cost + Math.min(Math.min(dtwMatrix[i - 1][j], dtwMatrix[i][j - 1]), dtwMatrix[i - 1][j - 1]);
            }
        }

        // 매칭 결과 저장할 리스트
        int[] matchedUserOnsetIndices = new int[n];
        Arrays.fill(matchedUserOnsetIndices, -1);
        boolean[] answerOnsetPlayed = new boolean[m];  // answerOnsetPlayed 배열 초기화 (m 크기)
        int i = n;  // i는 userOnset 크기
        int j = m;  // j는 answerOnset 크기
        while (i > 0 && j > 0) {
            double cost = Math.abs(userOnset.get(i - 1) - answerOnset.get(j - 1));  // 매칭된 비용 계산

            if (cost <= threshold) {  // threshold 이하일 경우
                matchedUserOnsetIndices[i - 1] = j - 1; // userOnset의 i-1번 온셋이 answerOnset의 j-1번 온셋과 매칭되었음을 표시
                answerOnsetPlayed[j - 1] = true;  // answerOnset에서 해당 음표가 연주되었음을 표시
                i--;
                j--;
            } else if (dtwMatrix[i - 1][j] <= Math.min(dtwMatrix[i][j - 1], dtwMatrix[i - 1][j - 1])) {
                i--;  // i 감소 (userOnset을 뒤로 이동)
            } else {
                j--;  // j 감소 (answerOnset을 뒤로 이동)
            }
        }

        // 결과 반환
        return OnsetMatchResult.builder()
                .userOnset(userOnset)
                .answerOnset(answerOnset)
                .answerOnsetPlayed(answerOnsetPlayed)
                .matchedUserOnsetIndices(matchedUserOnsetIndices)
                .build();
    }
}
