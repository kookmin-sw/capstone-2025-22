package com.capstone.practice.dto;

import com.capstone.practice.entity.SheetPractice;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SheetPracticeRepresentResponse {
    int userSheetId;
    String sheetName;
    LocalDateTime lastPracticeDate;
    int maxScore;

    public static SheetPracticeRepresentResponse from(UserSheet userSheet, int maxScore, LocalDateTime lastPracticeDate) {
        return SheetPracticeRepresentResponse.builder()
                .userSheetId(userSheet.getUserSheetId())
                .sheetName(userSheet.getSheetName())
                .lastPracticeDate(lastPracticeDate)
                .maxScore(maxScore)
                .build();
    }
}
