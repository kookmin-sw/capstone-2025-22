package com.capstone.letmedrum.common.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

@Builder
@Getter
@Setter
public class ApiResponse<T> extends ResponseEntity<CustomResponseDto<T>> {
    private T responseBody;
    private String message;
    private HttpStatus status;
    private ApiResponse(T body, String message, HttpStatus status) {
        super(new CustomResponseDto<T>(message, body), status);
    }
    /**
     * @param message String, error message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> error(String message) {
        return ApiResponse.<T>builder()
                .message(message)
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .build();
    }
    /**
     * @param message String, error message
     * @param status HttpStatus code
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> error(String message, HttpStatus status) {
        return ApiResponse.<T>builder()
                .message(message)
                .status(status)
                .build();
    }
    /**
     * @param body T, success response body
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> success(T body){
        return ApiResponse.<T>builder()
                .responseBody(body)
                .message(null)
                .status(HttpStatus.OK)
                .build();
    }
    /**
     * @param body T, success response body
     * @param message String, success message
     * @return ApiResponse - ApiResponse<T>, response dto
     * */
    public static <T> ApiResponse<T> success(T body, String message){
        return ApiResponse.<T>builder()
                .responseBody(body)
                .message(message)
                .status(HttpStatus.OK)
                .build();
    }
}
