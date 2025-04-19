package com.capstone.practice.dto;

import com.capstone.practice.entity.SheetPractice;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SheetPracticeResponseDto {
    int practiceId;
    LocalDateTime createdDate;
    int score;

    public static SheetPracticeResponseDto from(SheetPractice sheetPractice) {
        return SheetPracticeResponseDto.builder()
                .practiceId(sheetPractice.getSheetPracticeId())
                .createdDate(sheetPractice.getCreatedDate())
                .score(sheetPractice.getScore())
                .build();
    }
}
