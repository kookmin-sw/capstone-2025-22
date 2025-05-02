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
public class SheetCreateMeta {
    String sheetName;
    String color;
    boolean isOwner;
    String userEmail;
    String fileExtension;

    public UserSheet toUserSheetEntity(Sheet sheet){
        return UserSheet.builder()
                .sheetName(sheetName)
                .color(color)
                .isOwner(isOwner)
                .userEmail(userEmail)
                .sheet(sheet)
                .build();
    }
}