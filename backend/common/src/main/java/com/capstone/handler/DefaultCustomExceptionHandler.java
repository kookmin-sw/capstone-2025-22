package com.capstone.handler;

import com.capstone.exception.CustomException;
import com.capstone.response.ApiResponse;
import com.capstone.response.CustomResponseDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class DefaultCustomExceptionHandler {
    @ExceptionHandler({CustomException.class})
    public ResponseEntity<CustomResponseDto<Object>> getDefaultExceptionApiResponse(CustomException e){
        log.error(e.getMessage(), e);
        return ApiResponse.error(e.getMessage(), e.getStatus());
    }
}
