package com.capstone.letmedrum.user.dto.request;

import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;

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
    /**
     * convert UserCreateDto to User Entity with encoded password and default role(ROLE_USER)
     * @param passwordEncoder - PasswordEncoder, not-null
     * @return User - User, not-null */
    public User toEntityWithEncodedPassword(PasswordEncoder passwordEncoder){
        this.password = passwordEncoder.encode(this.password);
        return toEntity();
    }
}
