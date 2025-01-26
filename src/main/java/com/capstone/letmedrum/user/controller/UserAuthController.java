package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.ApiResponse;
import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.service.UserAuthService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
public class UserAuthController {
    private final UserAuthService userAuthService;
    public UserAuthController(UserAuthService userAuthService){
        this.userAuthService = userAuthService;
    }
    @PostMapping("/signin")
    public ApiResponse<UserAuthResponseDto> signInUser(@RequestBody UserAuthInfoDto userAuthInfoDto){
        if(!userAuthInfoDto.validateUserAuthInfoDto()) return ApiResponse.error("signIn fail : invalid request body");
        UserAuthResponseDto responseDto = userAuthService.signInUser(userAuthInfoDto);
        return ApiResponse.success(responseDto);
    }
    @PostMapping("/signup")
    public ApiResponse<UserAuthResponseDto> signUpUser(@RequestBody UserCreateDto userCreateDto){
        if(!userCreateDto.validateUserCreateDto()) return ApiResponse.error("signUp fail : invalid request body");
        UserAuthResponseDto responseDto = userAuthService.signUpUser(userCreateDto);
        return ApiResponse.success(responseDto);
    }
    @GetMapping("/test")
    public ApiResponse<String> testApi(){
        return ApiResponse.success("success", "success");
    }
}
