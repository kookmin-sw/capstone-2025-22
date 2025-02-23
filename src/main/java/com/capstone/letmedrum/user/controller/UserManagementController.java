package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.enums.SuccessFlag;
import com.capstone.letmedrum.common.dto.ApiResponse;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.request.UserPasswordUpdateDto;
import com.capstone.letmedrum.user.dto.request.UserProfileUpdateRequestDto;
import com.capstone.letmedrum.user.dto.response.UserProfileUpdateResponseDto;
import com.capstone.letmedrum.user.service.UserUpdateService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/users")
public class UserManagementController {
    private final UserUpdateService userUpdateService;
    public UserManagementController(UserUpdateService userUpdateService) {
        this.userUpdateService = userUpdateService;
    }
    @PutMapping("/password")
    public ApiResponse<String> updatePassword(@RequestBody UserPasswordUpdateDto updateDto) {
        userUpdateService.updatePassword(updateDto);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
    @PutMapping("/profile")
    public ApiResponse<UserProfileUpdateResponseDto> updateProfile(
            @RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String accessToken,
            @RequestBody UserProfileUpdateRequestDto updateDto) {
        UserProfileUpdateResponseDto res = userUpdateService.updateProfile(accessToken, updateDto);
        return ApiResponse.success(res);
    }
}
