package com.capstone.letmedrum.mail.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class AuthCodeTemplateDto {
    private String title;
    private String content;
    private String authCode;
}
