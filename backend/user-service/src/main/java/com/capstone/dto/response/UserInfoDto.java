package com.capstone.dto.response;

import com.capstone.dto.UserResponseDto;
import com.capstone.enums.UserRole;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserInfoDto {
    String email;
    String password;
    String nickname;
    UserRole role;
    public UserResponseDto toResponseDto(){
        return UserResponseDto.builder()
                .email(email)
                .password(password)
                .nickname(nickname)
                .role(role)
                .build();
    }
}
