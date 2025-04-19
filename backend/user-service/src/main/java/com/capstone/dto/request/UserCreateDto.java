package com.capstone.dto.request;

import com.capstone.entity.User;
import com.capstone.enums.UserRole;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class UserCreateDto {
    private String email;
    private String password;
    private String nickname;
    @Builder
    public UserCreateDto(String email, String password, String nickname) {
        this.email = email;
        this.password = password;
        this.nickname = nickname;
    }
    /**
     * if object is valid, return true
     * get access token from HttpServletRequest object
     * @return isValid - boolean, not-null */
    public boolean validate(){
        return email!=null && password!=null && nickname!=null;
    }
    /**
     * convert UserCreateDto to User Entity with default role (ROLE_USER)l
     * @return User - User, not-null */
    public User toEntity(){
        return User.builder()
                .email(email)
                .password(password)
                .nickname(nickname)
                .role(UserRole.ROLE_USER)
                .build();
    }
    /**
     * convert UserCreateDto to User Entity with custom role
     * @param role - UserRole, not-null
     * @return User - User, not-null */
    public User toEntity(UserRole role){
        return User.builder()
                .email(email)
                .password(password)
                .nickname(nickname)
                .role(role)
                .build();
    }
}
