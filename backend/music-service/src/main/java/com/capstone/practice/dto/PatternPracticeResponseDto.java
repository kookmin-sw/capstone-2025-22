package com.capstone.practice.dto;

import com.capstone.practice.entity.PatternPractice;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PatternPracticeResponseDto {
    Long practiceId;
    Long patternId;
    double score;
    String practiceInfo;

    public static PatternPracticeResponseDto from(PatternPractice patternPractice) {
        return PatternPracticeResponseDto.builder()
                .practiceId(patternPractice.getPatternPracticeId())
                .patternId(patternPractice.getPattern().getId())
                .score(Double.parseDouble(patternPractice.getScore()))
                .practiceInfo(patternPractice.getPracticeInfo())
                .build();
    }
}
