package com.capstone.dto.response;

import lombok.*;

@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class UserProfileUpdateResponseDto {
    private String profileImage;
    private String nickname;
}