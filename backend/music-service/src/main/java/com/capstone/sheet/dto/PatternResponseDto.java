package com.capstone.sheet.dto;

import com.capstone.sheet.entity.Pattern;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PatternResponseDto {
    Long patternId;
    String patternName;
    int score;
    byte[] patternInfo;

    public static PatternResponseDto from(Pattern pattern) {
        return PatternResponseDto.builder()
                .patternId(pattern.getId())
                .patternName(pattern.getPatternName())
                .patternInfo(pattern.getPatternInfo())
                .build();
    }

    public static PatternResponseDto from(Pattern pattern, int score) {
        return PatternResponseDto.builder()
                .patternId(pattern.getId())
                .patternName(pattern.getPatternName())
                .patternInfo(pattern.getPatternInfo())
                .score(score)
                .build();
    }
}
