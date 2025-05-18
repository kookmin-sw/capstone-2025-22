package com.capstone.practice.dto;

import com.capstone.dto.score.FinalMeasureResult;
import com.capstone.practice.entity.SheetPractice;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SheetPracticeDetailResponseDto {
    int practiceId;
    int sheetId;
    String sheetName;
    String artistName;
    LocalDateTime createdDate;
    int score;
    List<FinalMeasureResult> practiceInfo;

    public static SheetPracticeDetailResponseDto from(SheetPractice sheetPractice) {
        List<FinalMeasureResult> finalMeasureResultList = null;
        try {
            finalMeasureResultList = new ObjectMapper()
                    .readValue(sheetPractice.getPracticeInfo(), new TypeReference<>() {});
        }catch (Exception e){
            log.error(e.getMessage(), e);
        }
        return SheetPracticeDetailResponseDto.builder()
                .practiceId(sheetPractice.getSheetPracticeId())
                .sheetId(sheetPractice.getUserSheet().getUserSheetId())
                .sheetName(sheetPractice.getUserSheet().getSheetName())
                .artistName(sheetPractice.getUserSheet().getSheet().getAuthor())
                .createdDate(sheetPractice.getCreatedDate())
                .score(sheetPractice.getScore())
                .practiceInfo(finalMeasureResultList)
                .build();
    }
}
