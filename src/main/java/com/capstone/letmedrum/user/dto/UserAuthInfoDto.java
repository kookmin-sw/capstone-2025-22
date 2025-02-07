package com.capstone.letmedrum.user.dto;

import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
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
    public UserAuthInfoDto(User user){
        this(user.getEmail(), user.getRole());
    }
    public boolean validate(){
        return email != null && role != null;
    }
}
