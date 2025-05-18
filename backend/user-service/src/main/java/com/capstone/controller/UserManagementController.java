package com.capstone.controller;

import com.capstone.constants.AuthConstants;
import com.capstone.dto.UserResponseDto;
import com.capstone.dto.request.UserCreateDto;
import com.capstone.dto.request.UserPasswordUpdateDto;
import com.capstone.dto.response.UserProfileUpdateResponseDto;
import com.capstone.entity.User;
import com.capstone.enums.SuccessFlag;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.UserRetrieveService;
import com.capstone.service.UserUpdateService;
import io.swagger.v3.oas.annotations.Operation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Optional;

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
    @Operation(summary = "find user by email")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> findUserByEmail(@RequestParam("email") String email) {
        User user = userRetrieveService.getUserOrExceptionByEmail(email);
        return ApiResponse.success(user.toResponseDto());
    }

    @GetMapping("/nickname")
    @Operation(summary = "find user by nickname")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> findUserByNickname(@RequestParam("nickname") String nickname) {
        User user = userRetrieveService.getUserOrExceptionByNickname(nickname);
        return ApiResponse.success(user.toResponseDto());
    }

    @PostMapping("")
    @Operation(summary = "save user ( never use it!!!! )")
    public ResponseEntity<CustomResponseDto<UserResponseDto>> saveUser(@RequestBody UserCreateDto userCreateDto) {
        return ApiResponse.success(userUpdateService.createUser(userCreateDto).toResponseDto());
    }

    @PutMapping("/password")
    @Operation(summary = "update user's password")
    public ResponseEntity<CustomResponseDto<String>> updatePassword(@RequestBody UserPasswordUpdateDto updateDto) {
        userUpdateService.updatePassword(updateDto);
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }

    @PutMapping(value = "/profile", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "update user's profile")
    public ResponseEntity<CustomResponseDto<UserProfileUpdateResponseDto>> updateProfile(
            @RequestHeader(AuthConstants.ACCESS_TOKEN_HEADER_KEY) String accessToken,
            @RequestPart(value = "nickname") String nickname,
            @RequestPart(value = "profileImage", required = false) Optional<MultipartFile> profileImageOptional) {
        MultipartFile profileImage = profileImageOptional.orElse(null);
        return ApiResponse.success(
                userUpdateService.updateProfile(accessToken, nickname, profileImage)
        );
    }
}
