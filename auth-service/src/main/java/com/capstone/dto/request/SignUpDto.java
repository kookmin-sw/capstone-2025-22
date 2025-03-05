package com.capstone.dto.request;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SignUpDto {
    private String email;
    private String password;
    private String nickname;

    public boolean validate(){
        return email != null && password != null && nickname != null;
    }
}
