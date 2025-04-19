package com.capstone.exception;

import org.springframework.http.HttpStatus;

public class InvalidUserInfoException extends CustomException {
    public InvalidUserInfoException(String message) {
        super(HttpStatus.NOT_FOUND, message);
    }
}
