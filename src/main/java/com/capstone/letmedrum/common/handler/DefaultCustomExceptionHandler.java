package com.capstone.letmedrum.common.handler;

import com.capstone.letmedrum.common.dto.ApiResponse;
import com.capstone.letmedrum.common.exception.CustomException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class DefaultCustomExceptionHandler {
    @ExceptionHandler({CustomException.class})
    public ApiResponse<String> getDefaultExceptionApiResponse(CustomException e){
        log.error(e.getMessage(), e);
        return ApiResponse.error(e.getMessage(), e.getStatus());
    }
}
