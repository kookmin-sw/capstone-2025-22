package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.common.ApiResponse;
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
}
