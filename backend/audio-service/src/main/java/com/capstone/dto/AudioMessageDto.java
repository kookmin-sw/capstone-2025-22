package com.capstone.dto;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class AudioMessageDto {
    private int bpm;
    private int userSheetId;
    private String email;
    private String message;
    private String measureNumber;
}
