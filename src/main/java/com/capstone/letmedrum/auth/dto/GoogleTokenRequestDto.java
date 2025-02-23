package com.capstone.letmedrum.auth.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class GoogleTokenRequestDto {
    String client_id;
    String client_secret;
    String redirect_uri;
    String code;
    String grant_type;
}
