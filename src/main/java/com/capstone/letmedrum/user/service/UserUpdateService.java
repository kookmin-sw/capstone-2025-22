package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.exception.InvalidTokenException;
import com.capstone.letmedrum.auth.service.AuthManagerService;
import com.capstone.letmedrum.common.exception.InvalidRequestException;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.request.UserPasswordUpdateDto;
import com.capstone.letmedrum.user.dto.request.UserProfileUpdateRequestDto;
import com.capstone.letmedrum.user.dto.response.UserProfileUpdateResponseDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.exception.InvalidUserInfoException;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserUpdateService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final PasswordEncoder passwordEncoder;
    private final AuthManagerService authManagerService;
    public UserUpdateService(UserRepository userRepository, JwtUtils jwtUtils, PasswordEncoder passwordEncoder, AuthManagerService authManagerService) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
        this.authManagerService = authManagerService;
    }
    @Transactional
    public void updatePassword(UserPasswordUpdateDto updateDto) {
        if(!jwtUtils.validateToken(updateDto.getEmailToken())){
            throw new InvalidRequestException("Invalid email token : expired or invalid");
        }
        String email = jwtUtils.getUserEmail(updateDto.getEmailToken());
        if(!authManagerService.getEmailToken(email).equals(updateDto.getEmailToken())){
            throw new InvalidRequestException("Invalid email token : expired or invalid");
        }
        String encodedPassword = passwordEncoder.encode(updateDto.getNewPassword());
        User user = userRepository.findByEmail(email)
                .orElseThrow(()->new InvalidUserInfoException("User not found"));
        user.setPassword(encodedPassword);
    }
    @Transactional
    public UserProfileUpdateResponseDto updateProfile(String accessToken, UserProfileUpdateRequestDto updateDto) {
        accessToken = jwtUtils.processToken(accessToken);
        if(!jwtUtils.validateToken(accessToken)){
            throw new InvalidTokenException("Invalid access token : expired or invalid");
        }
        String email = jwtUtils.getUserEmail(accessToken);
        User user = userRepository.findByEmail(email)
                .orElseThrow(()->new InvalidUserInfoException("User not found"));
        user.setNickname(updateDto.getNickname());
        user.setProfileImage(updateDto.getProfileImage());
        return UserProfileUpdateResponseDto.builder()
                .nickname(updateDto.getNickname())
                .profileImage(updateDto.getProfileImage()).build();
    }
}
