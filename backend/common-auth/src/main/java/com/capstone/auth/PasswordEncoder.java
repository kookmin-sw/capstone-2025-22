package com.capstone.auth;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;

@Component
public class PasswordEncoder extends BCryptPasswordEncoder {
    public boolean isEncoded(String encodedPassword) {
        return encodedPassword.matches("^\\$2[ayb]\\$\\d{2}\\$[./A-Za-z0-9]{53}$");
    }
}
