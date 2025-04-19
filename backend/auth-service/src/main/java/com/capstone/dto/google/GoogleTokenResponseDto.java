package com.capstone.dto.google;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class GoogleTokenResponseDto {
    private int expires_id;
    private String access_token;
    private String token_type;
    private String expires_in;
    private String scope;
}
