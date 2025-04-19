package com.capstone.controller;

import com.capstone.dto.EmailTokenResponseDto;
import com.capstone.enums.SuccessFlag;
import com.capstone.exception.InvalidRequestException;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import com.capstone.service.UserInfoVerificationService;
import com.capstone.service.VerificationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/verification")
public class VerificationController {
    private final VerificationService verificationService;
    private final UserInfoVerificationService userVerificationService;
    /**
     * constructor for DI
     * @param verificationService UserVerificationService class
    * */
    public VerificationController(VerificationService verificationService, UserInfoVerificationService userVerificationService) {
        this.verificationService = verificationService;
        this.userVerificationService = userVerificationService;
    }
    /**
     * API to check if nickname is valid
     * @param nickname user's nickname
     * @return success if auth code is valid
     * */
    @GetMapping("/nicknames")
    public Mono<ResponseEntity<CustomResponseDto<String>>> checkNickname(@RequestParam(name="nickname") String nickname){
        return userVerificationService.isValidNickname(nickname)
                .map(res -> {
                    if(!res) return ApiResponse.success(SuccessFlag.FAILURE.getLabel());
                    return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
                });
    }
    /**
     * API to check if email is valid
     * @param email user's email
     * @return success if auth code is valid
     * */
    @GetMapping("/emails")
    public Mono<ResponseEntity<CustomResponseDto<String>>> checkEmail(@RequestParam(name="email") String email){
        return userVerificationService.isValidEmail(email)
                .map(res -> {
                    if(!res) return ApiResponse.success(SuccessFlag.FAILURE.getLabel());
                    return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
                });
    }
    /**
     * API to send auth mail to user
     * @param email user's email
     * @return success if auth code is valid
    * */
    @GetMapping("/auth-codes")
    public Mono<ResponseEntity<CustomResponseDto<String>>> sendAuthCodes(@RequestParam(name="email") String email){
        return verificationService.sendVerificationEmail(email).map(res -> {
            if(!res) return ApiResponse.success(SuccessFlag.FAILURE.getLabel());
            return ApiResponse.success(SuccessFlag.SUCCESS.getLabel());
        });
    }
    /**
     * API to check auth code
     * @param email user's email
     * @param authCode user's auth code
     * @return success if auth code is valid
    * */
    @GetMapping("/auth-codes/check")
    public Mono<ResponseEntity<CustomResponseDto<EmailTokenResponseDto>>> checkAuthCode(@RequestParam(name="email") String email, @RequestParam(name="authCode") String authCode){
        return verificationService.isValidAuthCode(email, authCode)
                .flatMap(res -> {
                    if(!res) return Mono.error(new InvalidRequestException("invalid email auth code"));
                    return verificationService.createEmailVerificationToken(email);
                }).map(emailToken -> ApiResponse.success(new EmailTokenResponseDto(emailToken)));
    }
}
