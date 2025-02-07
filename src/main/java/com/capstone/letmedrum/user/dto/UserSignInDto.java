package com.capstone.letmedrum.user.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Builder
@Getter
@Setter
public class UserSignInDto {
    private final String email;
    private final String password;
    public UserSignInDto(String email, String password) {
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
