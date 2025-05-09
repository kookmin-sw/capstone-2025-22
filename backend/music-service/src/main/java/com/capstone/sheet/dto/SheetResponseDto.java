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
    private int userSheetId;
    private String sheetName;
    private String color;
    private LocalDateTime lastPracticeDate;
    private LocalDateTime createdDate;

    public static SheetResponseDto from(UserSheet userSheet, LocalDateTime lastPracticeDate){
        return SheetResponseDto.builder()
                .userSheetId(userSheet.getUserSheetId())
                .sheetName(userSheet.getSheetName())
                .color(userSheet.getColor())
                .lastPracticeDate(lastPracticeDate)
                .createdDate(userSheet.getCreatedDate()).build();
    }

    public static SheetResponseDto from(UserSheet userSheet){
        return SheetResponseDto.builder()
                .userSheetId(userSheet.getUserSheetId())
                .sheetName(userSheet.getSheetName())
                .color(userSheet.getColor())
                .createdDate(userSheet.getCreatedDate()).build();
    }
}
