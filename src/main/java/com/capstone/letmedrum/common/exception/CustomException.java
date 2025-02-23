package com.capstone.letmedrum.common.exception;

import lombok.Getter;
import lombok.Setter;
import org.springframework.http.HttpStatus;

@Setter
@Getter
public class CustomException extends RuntimeException {
    private final HttpStatus status;
    public CustomException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }
}
