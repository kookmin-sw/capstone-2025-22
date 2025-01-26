package com.capstone.letmedrum.common;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
public class ApiResponse<T>{
    String message;
    T body;
    /**
     * @param message String, error message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> error(String message){
        return ApiResponse.<T>builder()
                .body(null)
                .message(message)
                .build();
    }
    /**
     * @param body T, success response body
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> success(T body){
        return ApiResponse.<T>builder()
                .body(body)
                .message(null)
                .build();
    }
    /**
     * @param body T, success response body
     * @param message String, success message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> success(T body, String message){
        return ApiResponse.<T>builder()
                .body(body)
                .message(message)
                .build();
    }
}
