package com.capstone.service;

import com.capstone.client.AuthClientService;
import com.capstone.dto.request.UserCreateDto;
import com.capstone.dto.request.UserPasswordUpdateDto;
import com.capstone.dto.request.UserProfileUpdateRequestDto;
import com.capstone.dto.response.UserInfoDto;
import com.capstone.dto.response.UserProfileUpdateResponseDto;
import com.capstone.auth.PasswordEncoder;
import com.capstone.entity.User;
import com.capstone.exception.CustomException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.exception.InvalidTokenException;
import com.capstone.exception.InvalidUserInfoException;
import com.capstone.jwt.JwtUtils;
import com.capstone.repository.UserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserUpdateService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final AuthClientService authClientService;
    private final PasswordEncoder passwordEncoder;
    public UserUpdateService(UserRepository userRepository, JwtUtils jwtUtils, AuthClientService authClientService, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.authClientService = authClientService;
        this.passwordEncoder = passwordEncoder;
    }
    @Transactional
    public void updatePassword(UserPasswordUpdateDto updateDto) {
        if(!jwtUtils.validateToken(updateDto.getEmailToken())){
            throw new InvalidRequestException("Invalid email token : expired or invalid");
        }
        String email = jwtUtils.getUserEmail(updateDto.getEmailToken());
        if(!authClientService.findEmailTokenSync(email).equals(updateDto.getEmailToken())){
            throw new InvalidTokenException("Invalid email token : expired or invalid");
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
    @Transactional
    public UserInfoDto createUser(UserCreateDto userCreateDto) {
        if(userRepository.findByEmail(userCreateDto.getEmail()).isPresent())
            throw new CustomException(HttpStatus.CONFLICT, "user already exists");
        if(userRepository.findByNickname(userCreateDto.getNickname()).isPresent())
            throw new CustomException(HttpStatus.CONFLICT, "nickname duplicated");
        User user = userRepository.save(userCreateDto.toEntity());
        return UserInfoDto.builder()
                .email(user.getEmail())
                .password(user.getPassword())
                .nickname(user.getNickname())
                .role(user.getRole())
                .build();
    }
}
