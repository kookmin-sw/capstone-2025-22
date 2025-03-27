package com.capstone.sheet.dto;

import lombok.*;

import java.util.List;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SheetListResponseDto {
    List<SheetResponseDto> sheets;
}
