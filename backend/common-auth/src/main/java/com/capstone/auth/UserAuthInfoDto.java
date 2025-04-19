package com.capstone.auth;

import com.capstone.enums.UserRole;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
public class UserAuthInfoDto {
    private String email;
    private UserRole role;
    public UserAuthInfoDto(String email, UserRole role){
        this.email = email;
        this.role = role;
    }
    public boolean validate(){
        return email != null && role != null;
    }
}
