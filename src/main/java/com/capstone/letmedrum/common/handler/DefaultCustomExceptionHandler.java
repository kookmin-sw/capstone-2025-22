package com.capstone.letmedrum.common.handler;

import com.capstone.letmedrum.common.dto.ApiResponse;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class DefaultCustomExceptionHandler {
    @ExceptionHandler({RuntimeException.class})
    public ApiResponse<String> getDefaultExceptionApiResponse(Exception e){
        return ApiResponse.error(e.getMessage());
    }
}
