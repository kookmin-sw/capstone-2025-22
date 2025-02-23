package com.capstone.letmedrum.user.dto.response;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@Getter
@NoArgsConstructor
public class UserEmailTokenResponseDto {
    public UserEmailTokenResponseDto(String emailToken) {
        this.emailToken = emailToken;
    }
    private String emailToken;
}
