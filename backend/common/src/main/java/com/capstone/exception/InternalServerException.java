package com.capstone.exception;

import org.springframework.http.HttpStatus;

public class InternalServerException extends CustomException {
    public InternalServerException(String message) {
        super(HttpStatus.INTERNAL_SERVER_ERROR, message);
    }
}
