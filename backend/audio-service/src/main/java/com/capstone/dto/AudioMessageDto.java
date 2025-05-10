package com.capstone.dto;

import lombok.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class AudioMessageDto {
    int bpm;
    int userSheetId;
    String identifier;
    String email;
    String message;
    String measureNumber;
    boolean endOfMeasure;
}
