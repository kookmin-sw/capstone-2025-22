package com.capstone.letmedrum.common.exception;

import org.springframework.http.HttpStatus;

public class InvalidRequestException extends CustomException{
    public InvalidRequestException(String message) {
        super(HttpStatus.BAD_REQUEST, message);
    }
}
