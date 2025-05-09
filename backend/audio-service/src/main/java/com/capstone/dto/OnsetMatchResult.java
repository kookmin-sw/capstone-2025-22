package com.capstone.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
public class OnsetMatchResult {
    List<Double> userOnset;
    List<Double> answerOnset;
    boolean[] answerOnsetPlayed;
    int[] matchedUserOnsetIndices;
}
