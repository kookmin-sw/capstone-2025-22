package com.capstone.dto.google;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class GoogleTokenRequestDto {
    String client_id;
    String client_secret;
    String redirect_uri;
    String code;
    String grant_type;
}
