package com.capstone.dto;

import com.capstone.enums.UserRole;
import lombok.*;

@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class UserResponseDto {
    String email;
    String password;
    String nickname;
    UserRole role;
}
