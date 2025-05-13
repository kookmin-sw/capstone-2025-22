package com.capstone.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PatternMessageDto {
    String audioBase64;
    String identifier;
    String email;
    String measureNumber;
    boolean endOfMeasure;
    Long patternId;
}
