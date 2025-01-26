package com.capstone.letmedrum.user.dto;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class UserAuthResponseDto {
    private final String email;
    private final String accessToken;
    private final String refreshToken;
    public UserAuthResponseDto(String email, String accessToken, String refreshToken){
        this.email = email;
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
    }
}
