package com.capstone.sheet.dto;

import com.capstone.sheet.entity.UserSheet;
import lombok.*;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SheetDetailResponseDto {
    int userSheetId;
    String sheetName;
    LocalDateTime lastPracticeDate;
    String color;
    String sheetInfo;

    public static SheetDetailResponseDto from(UserSheet userSheet, LocalDateTime lastPracticeDate) {
        return SheetDetailResponseDto.builder()
                .userSheetId(userSheet.getUserSheetId())
                .sheetName(userSheet.getSheetName())
                .color(userSheet.getColor())
                .lastPracticeDate(lastPracticeDate)
                .sheetInfo(userSheet.getSheet().getSheetInfo())
                .build();
    }
}
