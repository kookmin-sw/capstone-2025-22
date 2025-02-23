package com.capstone.letmedrum.user.exception;

import com.capstone.letmedrum.common.exception.CustomException;
import org.springframework.http.HttpStatus;

public class InvalidUserInfoException extends CustomException {
    public InvalidUserInfoException(String message) {
        super(HttpStatus.NOT_FOUND, message);
    }
}
