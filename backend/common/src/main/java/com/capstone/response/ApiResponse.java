package com.capstone.response;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

@Builder
@Getter
@Setter
public class ApiResponse<T> {
    private T responseBody;
    private String message;
    private HttpStatus status;
    private ApiResponse(T body, String message, HttpStatus status) {
        this.responseBody = body;
        this.message = message;
        this.status = status;
    }
    public ResponseEntity<CustomResponseDto<T>> toResponseEntity() {
        CustomResponseDto<T> dto = new CustomResponseDto<>(message, responseBody);
        return new ResponseEntity<>(dto, status);
    }
    /**
     * @param message String, error message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ResponseEntity<CustomResponseDto<T>> error(String message) {
        return ApiResponse.<T>builder()
                .message(message)
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .build().toResponseEntity();
    }
    /**
     * @param message String, error message
     * @param status HttpStatus code
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ResponseEntity<CustomResponseDto<T>> error(String message, HttpStatus status) {
        return ApiResponse.<T>builder()
                .message(message)
                .status(status)
                .build().toResponseEntity();
    }
    /**
     * @param body T, success response body
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ResponseEntity<CustomResponseDto<T>> success(T body){
        return ApiResponse.<T>builder()
                .responseBody(body)
                .message(null)
                .status(HttpStatus.OK)
                .build().toResponseEntity();
    }
    /**
     * @param body T, success response body
     * @param message String, success message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ResponseEntity<CustomResponseDto<T>> success(T body, String message){
        return ApiResponse.<T>builder()
                .responseBody(body)
                .message(message)
                .status(HttpStatus.OK)
                .build().toResponseEntity();
    }
}
