package com.capstone.practice.dto;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.sheet.dto.SheetResponseDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SheetPracticeDetailResponseDto {
    int practiceId;
    int sheetId;
    String sheetName;
    LocalDateTime createdDate;
    int score;
    String practiceInfo;

    public static SheetPracticeDetailResponseDto from(SheetPractice sheetPractice) {
        return SheetPracticeDetailResponseDto.builder()
                .practiceId(sheetPractice.getSheetPracticeId())
                .sheetId(sheetPractice.getUserSheet().getUserSheetId())
                .sheetName(sheetPractice.getUserSheet().getSheetName())
                .createdDate(sheetPractice.getCreatedDate())
                .score(sheetPractice.getScore())
                .practiceInfo(sheetPractice.getPracticeInfo())
                .build();
    }
}
