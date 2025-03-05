package com.capstone.controller;

import com.capstone.dto.request.AuthGoogleRequestDto;
import com.capstone.dto.response.AuthResponseDto;
import com.capstone.dto.request.SignInDto;
import com.capstone.dto.request.SignUpDto;
import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InvalidRequestException;
import com.capstone.exception.InvalidTokenException;
import com.capstone.jwt.JwtUtils;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.UserAuthService;
import com.capstone.service.UserGoogleAuthService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

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
    public ResponseEntity<CustomResponseDto<String>> testApi(){
        return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
    }
    /**
     * 이메일과 비밀번호를 활용하여 사용자 로그인 처리하는 API
     * @param signInDto - 로그인 요청 객체
     * @return access token과 refresh token을 포함한 응답객체
    * */
    @PostMapping("/signin")
    public Mono<ResponseEntity<CustomResponseDto<AuthResponseDto>>> signInUser(@RequestBody SignInDto signInDto){
        if(!signInDto.validate()) throw new InvalidRequestException("invalid request body");
        return userAuthService.signInUser(signInDto)
                .map(ApiResponse::success);
    }
    /*
    * */
    @PostMapping("/signin/google")
    public Mono<ResponseEntity<CustomResponseDto<AuthResponseDto>>> signInGoogle(@RequestBody AuthGoogleRequestDto authGoogleRequestDto){
        if(authGoogleRequestDto.getGoogleAuthCode()==null) throw new InvalidRequestException("invalid request body");
        return userGoogleAuthService.signIn(authGoogleRequestDto.getGoogleAuthCode())
                .map(ApiResponse::success);
    }
    /**
     * 아직 가입되지 않은 사용자를 회원가입 처리하는 API
     * @param signUpDto - 사용자 회원가입 요청 객체
     * @return access token과 refresh token을 포함한 응답객체
    * */
    @PostMapping("/signup")
    public Mono<ResponseEntity<CustomResponseDto<AuthResponseDto>>> signUpUser(@RequestBody SignUpDto signUpDto){
        if(!signUpDto.validate()) throw new InvalidRequestException("invalid request body");
        return userAuthService.signUpUser(signUpDto)
                .map(ApiResponse::success);
    }
    /**
     * 토큰 무효화를 통한 로그아웃 처리
     * @param refreshToken - 사용자 refresh token
     * @return 성공여부
    * */
    @GetMapping("/signout")
    public Mono<ResponseEntity<CustomResponseDto<String>>> signOutUser(@RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String refreshToken){
        refreshToken = jwtUtils.processToken(refreshToken);
        return userAuthService.signOutUser(refreshToken)
                .flatMap(res -> {
                    if(!res) return Mono.error(new InvalidTokenException("invalid refresh token"));
                    return Mono.just(ApiResponse.success(SuccessFlag.SUCCESS.getLabel()));
                });
    }
    /**
     * 사용자의 refresh token을 기반으로, 유효한 토큰이라면 access token과 refresh token을 재발급하는 API
     * @param refreshToken - 사용자 refresh token
     * @return access token과 refresh token을 포함한 응답객체
     * */
    @PostMapping("/token")
    public Mono<ResponseEntity<CustomResponseDto<AuthResponseDto>>> regenerateToken(@RequestHeader(JwtUtils.ACCESS_TOKEN_HEADER_KEY) String refreshToken){
        refreshToken = jwtUtils.processToken(refreshToken);
        if(refreshToken==null || refreshToken.isEmpty()) throw new InvalidTokenException("token is null or empty");
        return userAuthService.doRefreshTokenRotation(refreshToken)
                .map(ApiResponse::success);
    }
}
