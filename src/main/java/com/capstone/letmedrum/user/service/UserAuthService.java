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
    /**
     * Check user's id and password and return auth info
     * @param userAuthInfoDto UserAuthInfoDto, not-null
     * @return UserAuthResponseDto - not null
     * @throws RuntimeException If user info is not valid
    * */
    public UserAuthResponseDto signInUser(UserAuthInfoDto userAuthInfoDto){
        User existUser = userRepository.findByEmail(userAuthInfoDto.getEmail())
                .orElseThrow(() -> new RuntimeException("sign in error : invalid email"));
        if(!passwordEncoder.matches(userAuthInfoDto.getPassword(), existUser.getPassword())){
            throw new RuntimeException("sign in error : invalid password");
        }
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(userAuthInfoDto.getEmail())
                .build();
    }
    /**
     * Sign up the user if a user with that email does not exist.
     * @param userCreateDto UserCreateDto not-null
     * @return UserAuthResponseDto - not null
     * @throws RuntimeException If user already exists
    * */
    @Transactional
    public UserAuthResponseDto signUpUser(UserCreateDto userCreateDto){
        userRepository.findByEmail(userCreateDto.getEmail())
                .ifPresent(user -> {
                    throw new RuntimeException("sign up error : user already exists : " + userCreateDto.getEmail());
                });
        User user = userRepository.save(userCreateDto.toEntityWithEncodedPassword(passwordEncoder));
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(user.getEmail())
                .password(user.getPassword())
                .role(user.getRole())
                .build();
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(user.getEmail())
                .build();
    }
}
