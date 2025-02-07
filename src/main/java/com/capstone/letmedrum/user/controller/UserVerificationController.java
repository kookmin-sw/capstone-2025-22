package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.SuccessFlag;
import com.capstone.letmedrum.common.dto.ApiResponse;
import com.capstone.letmedrum.user.service.UserRetrieveService;
import com.capstone.letmedrum.user.service.UserVerificationService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/verification")
public class UserVerificationController {
    private final UserRetrieveService userRetrieveService;
    private final UserVerificationService userVerificationService;
    /**
     * constructor for DI
     * @param userRetrieveService UserRetrieveService class
     * @param userVerificationService UserVerificationService class
    * */
    public UserVerificationController(UserRetrieveService userRetrieveService, UserVerificationService userVerificationService) {
        this.userRetrieveService = userRetrieveService;
        this.userVerificationService = userVerificationService;
    }
    /**
     * API to check if nickname is valid
     * @param nickname user's nickname
     * @return success if auth code is valid
     * */
    @GetMapping("/nicknames")
    public ApiResponse<String> checkNickname(@RequestParam(name="nickname") String nickname){
        if(userRetrieveService.getUserOrNullByNickname(nickname) != null){
            return ApiResponse.success(SuccessFlag.FAILURE.name());
        }
        return ApiResponse.success(SuccessFlag.SUCCESS.name());
    }
    /**
     * API to check if email is valid
     * @param email user's email
     * @return success if auth code is valid
     * */
    @GetMapping("/emails")
    public ApiResponse<String> checkEmail(@RequestParam(name="email") String email){
        if(userRetrieveService.getUserOrNullByEmail(email) != null){
            return ApiResponse.success(SuccessFlag.FAILURE.name());
        }
        return ApiResponse.success(SuccessFlag.SUCCESS.name());
    }
    /**
     * API to send auth mail to user
     * @param email user's email
     * @return success if auth code is valid
    * */
    @GetMapping("/auth-codes")
    public ApiResponse<String> sendAuthCodes(@RequestParam(name="email") String email){
        return userVerificationService.sendVerificationEmail(email) ?
                ApiResponse.success(SuccessFlag.SUCCESS.name()) : ApiResponse.success(SuccessFlag.FAILURE.name());
    }
    /**
     * API to check auth code
     * @param email user's email
     * @param authCode user's auth code
     * @return success if auth code is valid
    * */
    @GetMapping("/auth-codes/check")
    public ApiResponse<String> checkAuthCode(@RequestParam(name="email") String email, @RequestParam(name="authCode") String authCode){
        return userVerificationService.isValidAuthCode(email, authCode) ?
                ApiResponse.success(SuccessFlag.SUCCESS.name()) : ApiResponse.success(SuccessFlag.FAILURE.name());
    }
}
