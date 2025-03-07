package com.capstone.controller;

import com.capstone.dto.UserResponseDto;
import com.capstone.dto.request.UserCreateDto;
import com.capstone.dto.request.UserPasswordUpdateDto;
import com.capstone.dto.request.UserProfileUpdateRequestDto;
import com.capstone.dto.response.UserProfileUpdateResponseDto;
import com.capstone.entity.User;
import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InternalServerException;
import com.capstone.jwt.JwtUtils;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.UserRetrieveService;
import com.capstone.service.UserUpdateService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
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
    @PutMapping(value = "/profile", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<ResponseEntity<CustomResponseDto<UserProfileUpdateResponseDto>>> updateProfile(
            @RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String accessToken,
            @RequestPart(value = "nickname") String nickname,
            @RequestPart(value = "profileImage", required = false) Mono<FilePart> profileImageMono) {
        return profileImageMono
                .flatMap(profileImage -> {
                    if(profileImage != null){
                        return userUpdateService.updateProfile(accessToken, nickname, profileImage);
                    } else {
                        // 파일이 없는 경우, null 전달
                        return userUpdateService.updateProfile(accessToken, nickname, null);
                    }
                })
                .switchIfEmpty(userUpdateService.updateProfile(accessToken, nickname, null))
                .map(ApiResponse::success)
                .onErrorResume(e -> {
                    // 예외 처리 로직 추가 (예: 로그 기록)
                    if (e instanceof ClassCastException) {
                        log.error(e.getMessage());
                        return Mono.error(new InternalServerException("Invalid file format."));
                    }
                    return Mono.error(new InternalServerException(e.getMessage()));
                });
    }
}
