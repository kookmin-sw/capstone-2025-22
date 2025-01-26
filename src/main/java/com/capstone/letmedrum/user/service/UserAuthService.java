package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserAuthService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final PasswordEncoder passwordEncoder;
    public UserAuthService(UserRepository userRepository, JwtUtils jwtUtils, PasswordEncoder passwordEncoder){
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
    }
    public UserAuthResponseDto signInUser(UserAuthInfoDto userAuthInfoDto){
        User existUser = userRepository.findByEmail(userAuthInfoDto.getEmail()).orElse(null);
        if(existUser==null) return null;
        if(!passwordEncoder.matches(userAuthInfoDto.getPassword(), existUser.getPassword())) return null;
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(userAuthInfoDto.getEmail())
                .build();
    }
    @Transactional
    public UserAuthResponseDto signUpUser(UserCreateDto userCreateDto){
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(userCreateDto.getEmail())
                .password(userCreateDto.getPassword())
                .role(userCreateDto.getRole())
                .build();
        User existUser = userRepository.findByEmail(userCreateDto.getEmail()).orElse(null);
        if(existUser!=null) return null;
        userRepository.save(userCreateDto.toEntityWithEncodedPassword(passwordEncoder));
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(userAuthInfoDto.getEmail())
                .build();
    }
}
