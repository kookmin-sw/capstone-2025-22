package com.capstone.letmedrum.mail.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Builder
@Getter
@Setter
@AllArgsConstructor
public class MailDto {
    private String from;
    private String to;
    private String text;
}
