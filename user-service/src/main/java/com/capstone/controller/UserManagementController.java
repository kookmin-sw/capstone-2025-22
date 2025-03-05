package com.capstone.controller;

import com.capstone.dto.UserResponseDto;
import com.capstone.dto.request.UserCreateDto;
import com.capstone.dto.request.UserPasswordUpdateDto;
import com.capstone.dto.request.UserProfileUpdateRequestDto;
import com.capstone.dto.response.UserProfileUpdateResponseDto;
import com.capstone.entity.User;
import com.capstone.enums.SuccessFlag;
import com.capstone.jwt.JwtUtils;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.UserRetrieveService;
import com.capstone.service.UserUpdateService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@Slf4j
@RestController
@RequestMapping("/users")
public class UserManagementController {
    private final UserUpdateService userUpdateService;
    private final UserRetrieveService userRetrieveService;
    public UserManagementController(UserUpdateService userUpdateService, UserRetrieveService userRetrieveService) {
        this.userUpdateService = userUpdateService;
        this.userRetrieveService = userRetrieveService;
    }
    @GetMapping("/email")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> findUserByEmail(@RequestParam("email") String email) {
        User user = userRetrieveService.getUserOrExceptionByEmail(email);
        return ApiResponse.success(user.toResponseDto());
    }
    @GetMapping("/nickname")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> findUserByNickname(@RequestParam("nickname") String nickname) {
        User user = userRetrieveService.getUserOrExceptionByNickname(nickname);
        return ApiResponse.success(user.toResponseDto());
    }
    @PostMapping("")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> saveUser(@RequestBody UserCreateDto userCreateDto) {
        return ApiResponse.success(userUpdateService.createUser(userCreateDto).toResponseDto());
    }
    @PutMapping("/password")
    public ResponseEntity<CustomResponseDto<String>> updatePassword(@RequestBody UserPasswordUpdateDto updateDto) {
        userUpdateService.updatePassword(updateDto);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
    @PutMapping("/profile")
    public ResponseEntity<CustomResponseDto<UserProfileUpdateResponseDto>> updateProfile(
            @RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String accessToken,
            @RequestBody UserProfileUpdateRequestDto updateDto) {
        UserProfileUpdateResponseDto res = userUpdateService.updateProfile(accessToken, updateDto);
        return ApiResponse.success(res);
    }
}
