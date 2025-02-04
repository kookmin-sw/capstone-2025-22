package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    private UserRepository userRepository;
    public UserService(UserRepository userRepository){
        this.userRepository = userRepository;
    }
    public boolean isValidNickname(String nickname){
        return true;
    }
}
