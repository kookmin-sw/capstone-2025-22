package com.capstone.letmedrum.auth.service;

import com.capstone.letmedrum.auth.exception.InvalidTokenException;
import com.capstone.letmedrum.common.exception.CustomException;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.auth.dto.UserAuthInfoDto;
import com.capstone.letmedrum.auth.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.request.UserCreateDto;
import com.capstone.letmedrum.auth.dto.UserSignInDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.exception.InvalidUserInfoException;
import com.capstone.letmedrum.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserAuthService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final PasswordEncoder passwordEncoder;
    private final AuthManagerService authManagerService;
    public UserAuthService(
            UserRepository userRepository,
            JwtUtils jwtUtils,
            PasswordEncoder passwordEncoder,
            AuthManagerService authManagerService
    ) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
        this.authManagerService = authManagerService;
    }
    /**
     * Check user's id and password and return auth info
     * @param userSignInDto UserSignInDto, not-null
     * @return UserAuthResponseDto - not null
     * @throws InvalidUserInfoException If user info is not valid
    * */
    public UserAuthResponseDto signInUser(UserSignInDto userSignInDto) {
        User existUser = userRepository.findByEmail(userSignInDto.getEmail()).orElse(null);
        if(!(existUser != null && passwordEncoder.matches(userSignInDto.getPassword(), existUser.getPassword()))){
            throw new InvalidUserInfoException("invalid email or password");
        }
        return generateResponseAndSaveToken(userSignInDto.getEmail(), UserRole.ROLE_USER);
    }
    /**
     * Sign up the user if a user with that email does not exist.
     * @param userCreateDto UserCreateDto not-null
     * @return UserAuthResponseDto - not null
     * @throws CustomException If user already exists
    * */
    @Transactional
    public UserAuthResponseDto signUpUser(UserCreateDto userCreateDto){
        userRepository.findByEmail(userCreateDto.getEmail())
                .ifPresent(user -> {
                    throw new CustomException(HttpStatus.CONFLICT, "user already exists");
                });
        userRepository.save(userCreateDto.toEntityWithEncodedPassword(passwordEncoder));
        return generateResponseAndSaveToken(userCreateDto.getEmail(), UserRole.ROLE_USER);
    }
    /**
     * sign out user by deleting token
     * @param refreshToken - user's refresh token
     * @return isSuccess - boolean, nullable
     * @throws InvalidTokenException if token is invalid
     * @throws InvalidUserInfoException if user not exists
    * */
    @Transactional
    public boolean signOutUser(String refreshToken) {
        if(!jwtUtils.validateToken(refreshToken)){
            throw new InvalidTokenException("invalid refresh token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        userRepository.findByEmail(email).orElseThrow(() -> new InvalidUserInfoException("user not found"));
        authManagerService.deleteAccessTokenAndRefreshToken(email);
        return true;
    }
    /**
     * regenerate access token and refresh token
     * @param refreshToken - user's refresh token, String
     * @return UserAuthResponseDto - UserAuthResponseDto, not-null
     * @throws InvalidTokenException if prev version token or invalid token is sent
    * */
    @Transactional
    public UserAuthResponseDto doRefreshTokenRotation(String refreshToken){
        if(!jwtUtils.validateToken(refreshToken)){
            throw new InvalidTokenException("invalid refresh token : invalid token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        String savedRefreshToken = authManagerService.getRefreshToken(email);
        if(!savedRefreshToken.equals(refreshToken)){
            authManagerService.deleteAccessTokenAndRefreshToken(email);
            throw new InvalidTokenException("invalid refresh token : using prev token");
        }
        return generateResponseAndSaveToken(email, UserRole.ROLE_USER);
    }
    /**
     * save tokens on redis and return user auth response
     * @param email user's email
     * @param role user's role
    * */
    public UserAuthResponseDto generateResponseAndSaveToken(String email, UserRole role){
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(email)
                .role(role)
                .build();
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        authManagerService.saveAccessTokenAndRefreshToken(email, accessToken, refreshToken);
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(email)
                .build();
    }
}
