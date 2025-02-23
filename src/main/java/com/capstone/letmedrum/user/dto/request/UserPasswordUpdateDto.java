package com.capstone.letmedrum.user.dto.request;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class UserPasswordUpdateDto {
    private String newPassword;
    private String emailToken;
}
