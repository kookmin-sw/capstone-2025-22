package com.capstone.sheet.dto;

import lombok.*;

import java.util.List;

@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class SheetListRequestDto {
    public List<Integer> sheetIds;
}
