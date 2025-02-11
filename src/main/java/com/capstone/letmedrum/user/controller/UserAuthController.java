package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.dto.ApiResponse;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.UserAuthGoogleRequestDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.dto.UserSignInDto;
import com.capstone.letmedrum.user.service.UserAuthService;
import com.capstone.letmedrum.user.service.UserGoogleAuthService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/auth")
public class UserAuthController {
    private final UserAuthService userAuthService;
    private final UserGoogleAuthService userGoogleAuthService;
    private final JwtUtils jwtUtils;

    public UserAuthController(UserAuthService userAuthService, UserGoogleAuthService userGoogleAuthService, JwtUtils jwtUtils) {
        this.userAuthService = userAuthService;
        this.userGoogleAuthService = userGoogleAuthService;
        this.jwtUtils = jwtUtils;
    }
    /**
     * 현재 서버가 요청을 정상적으로 받는지 확인하기 위한 API
     * @return - 성공여부
    * */
    @GetMapping("/test")
    public ApiResponse<String> testApi(){
        return ApiResponse.success("success", "success");
    }
    /**
     * 이메일과 비밀번호를 활용하여 사용자 로그인 처리하는 API
     * @param userSignInDto - 로그인 요청 객체
     * @return access token과 refresh token을 포함한 응답객체
    * */
    @PostMapping("/signin")
    public ApiResponse<UserAuthResponseDto> signInUser(@RequestBody UserSignInDto userSignInDto){
        if(!userSignInDto.validate()) return ApiResponse.error("signIn fail : invalid request body");
        UserAuthResponseDto responseDto = userAuthService.signInUser(userSignInDto);
        return ApiResponse.success(responseDto);
    }
    /*
    * */
    @PostMapping("/signin/google")
    public ApiResponse<UserAuthResponseDto> signInGoogle(@RequestBody UserAuthGoogleRequestDto userAuthGoogleRequestDto){
        if(userAuthGoogleRequestDto.getGoogleAuthCode()==null) return ApiResponse.error("signIn fail : invalid request body");
        return ApiResponse.success(userGoogleAuthService.signIn(userAuthGoogleRequestDto.getGoogleAuthCode()));
    }
    /**
     * 아직 가입되지 않은 사용자를 회원가입 처리하는 API
     * @param userCreateDto - 사용자 회원가입 요청 객체
     * @return access token과 refresh token을 포함한 응답객체
    * */
    @PostMapping("/signup")
    public ApiResponse<UserAuthResponseDto> signUpUser(@RequestBody UserCreateDto userCreateDto){
        if(!userCreateDto.validate()) return ApiResponse.error("signUp fail : invalid request body");
        UserAuthResponseDto responseDto = userAuthService.signUpUser(userCreateDto);
        return ApiResponse.success(responseDto);
    }
    /**
     * 토큰 무효화를 통한 로그아웃 처리
     * @param refreshToken - 사용자 refresh token
     * @return 성공여부
    * */
    @GetMapping("/signout")
    public ApiResponse<String> signOutUser(@RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String refreshToken){
        refreshToken = jwtUtils.processToken(refreshToken);
        boolean result = userAuthService.signOutUser(refreshToken);
        if(!result) return ApiResponse.error("signOut fail : invalid refresh token");
        return ApiResponse.success("success");
    }
    /**
     * 사용자의 refresh token을 기반으로, 유효한 토큰이라면 access token과 refresh token을 재발급하는 API
     * @param refreshToken - 사용자 refresh token
     * @return access token과 refresh token을 포함한 응답객체
     * */
    @PostMapping("/token")
    public ApiResponse<UserAuthResponseDto> regenerateToken(@RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String refreshToken){
        refreshToken = jwtUtils.processToken(refreshToken);
        if(refreshToken==null || refreshToken.isEmpty()) return ApiResponse.error("refresh token is null or empty");
        return ApiResponse.success(userAuthService.doRefreshTokenRotation(refreshToken));
    }
}
