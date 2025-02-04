package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.dto.ApiResponse;
import org.springframework.data.repository.query.Param;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/user")
public class UserController {
    @GetMapping("/test")
    public ApiResponse<String> testAuth(){
        return ApiResponse.success("success");
    }
    @GetMapping("/nickname-check/${nickname}")
    public ApiResponse<String> checkNickname(@Param("nickname") String nickname){
        return ApiResponse.success("nickname-check");
    }
}
