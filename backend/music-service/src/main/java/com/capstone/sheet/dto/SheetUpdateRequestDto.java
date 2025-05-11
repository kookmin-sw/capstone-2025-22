package com.capstone.sheet.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SheetUpdateRequestDto {
    @JsonProperty("email")
    String email;

    @JsonProperty("color")
    String color;

    @JsonProperty("name")
    String name;
}
