package com.capstone.sheet.dto;

import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SheetUpdateRequestDto {
    String email;
    String color;
    String name;
}
