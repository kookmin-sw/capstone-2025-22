package com.capstone.letmedrum.user.dto;

import com.capstone.letmedrum.user.entity.UserRole;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
public class UserAuthInfoDto {
    private String email;
    private String password;
    private UserRole role;
    public UserAuthInfoDto(String email, String password, UserRole role){
        this.email = email;
        this.password = password;
        this.role = role;
    }
    public UserAuthInfoDto(String email, String password){
        this(email, password, UserRole.ROLE_GUEST);
    }
    public boolean validateUserAuthInfoDto(){
        return email != null && password != null && role != null;
    }
}
