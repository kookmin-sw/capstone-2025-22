package com.capstone.dto.request;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Builder
@Getter
@Setter
@NoArgsConstructor
public class SignInDto {
    private String email;
    private String password;
    public SignInDto(String email, String password) {
        this.email = email;
        this.password = password;
    }
    /**
     * if object is valid, return true
     * @return iSValid - boolean, not-null
    * */
    public boolean validate() {
        return email != null && password != null;
    }
}
