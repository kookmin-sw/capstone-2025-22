package com.capstone.exception;

import org.springframework.http.HttpStatus;

public class DataNotFoundException extends CustomException{
    public DataNotFoundException(String message) {
        super(HttpStatus.NOT_FOUND, message);
    }
}
