package com.capstone.entity;

import com.capstone.dto.UserResponseDto;
import com.capstone.enums.UserRole;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long userId;
    @Column(name = "email")
    private String email;
    @Column(name = "password")
    private String password;
    @Column(name = "nickname")
    private String nickname;
    @Column(name = "role")
    private UserRole role;
    @Column(name="profile_image")
    private String profileImage;
    @Builder
    public User(String email, String password, String nickname, UserRole role) {
        this.email = email;
        this.password = password;
        this.nickname = nickname;
        this.role = role;
    }
    public UserResponseDto toResponseDto() {
        return UserResponseDto.builder()
                .email(this.email)
                .password(this.password)
                .nickname(this.nickname)
                .role(this.role)
                .build();
    }
}
