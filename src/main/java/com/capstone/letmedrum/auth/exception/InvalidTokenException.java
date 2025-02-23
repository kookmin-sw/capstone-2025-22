package com.capstone.letmedrum.auth.exception;

import com.capstone.letmedrum.common.exception.CustomException;
import org.springframework.http.HttpStatus;

public class InvalidTokenException extends CustomException {
    public InvalidTokenException(String message) {
        super(HttpStatus.UNAUTHORIZED, message);
    }
}
