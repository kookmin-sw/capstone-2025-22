package com.capstone.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@Getter
@NoArgsConstructor
public class EmailTokenResponseDto {
    public EmailTokenResponseDto(String emailToken) {
        this.emailToken = emailToken;
    }
    private String emailToken;
}
