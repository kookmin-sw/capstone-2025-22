package com.capstone.sheet.dto;

import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SheetCreateRequestDto {
    String sheetInfo;
    String author;
    String sheetName;
    String userEmail;
    String color;

    public Sheet toSheet() {
        return Sheet.builder()
                .sheetInfo(sheetInfo)
                .author(author)
                .build();
    }

    public UserSheet toUserSheet(Sheet sheet) {
        return UserSheet.builder()
                .sheet(sheet)
                .color(color)
                .sheetName(sheetName)
                .isOwner(true)
                .userEmail(userEmail).build();
    }
}
