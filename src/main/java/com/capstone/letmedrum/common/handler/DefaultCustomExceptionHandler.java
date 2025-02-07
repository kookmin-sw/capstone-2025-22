package com.capstone.letmedrum.common.handler;

import com.capstone.letmedrum.common.dto.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class DefaultCustomExceptionHandler {
    @ExceptionHandler({RuntimeException.class})
    public ApiResponse<String> getDefaultExceptionApiResponse(Exception e){
        log.error(e.getMessage(), e);
        return ApiResponse.error(e.getMessage());
    }
}
