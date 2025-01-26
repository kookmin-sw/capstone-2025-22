package com.capstone.letmedrum.user.dto;

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
    private UserRole role;
    @Builder
    public UserCreateDto(String email, String password, String nickname, UserRole role) {
        this.email = email;
        this.password = password;
        this.nickname = nickname;
        this.role = role;
    }
    public UserCreateDto(String email, String password, String name){
        this(email, password, name, UserRole.ROLE_USER);
    }

    public boolean validateUserCreateDto(){
        return email!=null && password!=null && nickname!=null && role!=null;
    }
    public User toEntity(){
        return User.builder()
                .email(email)
                .password(password)
                .nickname(nickname)
                .role(role)
                .build();
    }

    public User toEntityWithEncodedPassword(PasswordEncoder passwordEncoder){
        this.password = passwordEncoder.encode(this.password);
        return toEntity();
    }
}
