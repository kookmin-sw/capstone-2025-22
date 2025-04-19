package com.capstone.entity;

import com.capstone.dto.UserResponseDto;
import com.capstone.enums.UserRole;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.sql.Blob;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@EntityListeners(AuditingEntityListener.class)
public class User {
    @Id
    @Column(unique = true, nullable = false)
    private String email;
    @Column
    private String password;
    @Column(unique = true)
    private String nickname;
    @Column
    private UserRole role;
    @CreatedDate
    private LocalDateTime createdDate;
    @LastModifiedDate
    private LocalDateTime updatedDate;
    @Lob
    @Column(name="profile_image", columnDefinition = "Blob")
    private byte[] profileImage;
    public UserResponseDto toResponseDto() {
        return UserResponseDto.builder()
                .email(this.email)
                .password(this.password)
                .nickname(this.nickname)
                .role(this.role)
                .build();
    }
}
