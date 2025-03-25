package com.capstone.sheet.dto;


import com.capstone.sheet.entity.UserSheet;
import lombok.*;

import java.time.LocalDateTime;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class SheetResponseDto {
    private int sheetId;
    private String sheetName;
    private String color;
    private LocalDateTime lastPracticeDate;

    public static SheetResponseDto from(UserSheet userSheet, LocalDateTime lastPracticeDate){
        return SheetResponseDto.builder()
                .sheetId(userSheet.getSheetId())
                .sheetName(userSheet.getSheetName())
                .color(userSheet.getColor())
                .lastPracticeDate(lastPracticeDate).build();
    }

    public static SheetResponseDto from(UserSheet userSheet){
        return SheetResponseDto.builder()
                .sheetId(userSheet.getSheetId())
                .sheetName(userSheet.getSheetName())
                .color(userSheet.getColor()).build();
    }
}
