package com.capstone.sheet.dto;

import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import lombok.Data;

@Data
public class SheetCreateMeta {
    String sheetName;
    String color;
    boolean isOwner;
    String userEmail;
    String fileExtension;

    public Sheet toSheetEntity(byte[] sheetInfo){
        return Sheet.builder()
                .sheetInfo(sheetInfo)
                .build();
    }

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