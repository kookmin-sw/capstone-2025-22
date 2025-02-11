package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.dto.UserSignInDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UserAuthService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final PasswordEncoder passwordEncoder;
    private final AuthTokenService authTokenService;
    public UserAuthService(
            UserRepository userRepository,
            JwtUtils jwtUtils,
            PasswordEncoder passwordEncoder,
            AuthTokenService authTokenService
    ) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
        this.authTokenService = authTokenService;
    }
    /**
     * Check user's id and password and return auth info
     * @param userSignInDto UserSignInDto, not-null
     * @return UserAuthResponseDto - not null
     * @throws RuntimeException If user info is not valid
    * */
    public UserAuthResponseDto signInUser(UserSignInDto userSignInDto) {
        if(!verifyUserSignInInfo(userSignInDto)){
            throw new RuntimeException("invalid user info : email or password invalid");
        }
        return generateUserAuthResponseDto(userSignInDto.getEmail(), UserRole.ROLE_USER);
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
        userRepository.save(userCreateDto.toEntityWithEncodedPassword(passwordEncoder));
        return generateUserAuthResponseDto(userCreateDto.getEmail(), UserRole.ROLE_USER);
    }
    /**
     * sign out user by deleting token
     * @param refreshToken - user's refresh token
     * @return isSuccess - boolean, nullable
     * @throws RuntimeException - if refresh token is invalid
    * */
    @Transactional
    public boolean signOutUser(String refreshToken) {
        if(!jwtUtils.validateToken(refreshToken)){
            throw new RuntimeException("refresh token invalid : using invalid or expired token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        authTokenService.deleteAccessTokenAndRefreshToken(email);
        return true;
    }
    /**
     * regenerate access token and refresh token
     * @param refreshToken - user's refresh token, String
     * @return UserAuthResponseDto - UserAuthResponseDto, not-null
     * @throws RuntimeException - if prev version token or invalid token is sent
    * */
    @Transactional
    public UserAuthResponseDto doRefreshTokenRotation(String refreshToken){
        if(!jwtUtils.validateToken(refreshToken)){
            throw new RuntimeException("refresh token invalid : using invalid or expired token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        String savedRefreshToken = authTokenService.getRefreshToken(email);
        if(!savedRefreshToken.equals(refreshToken)){
            authTokenService.deleteAccessTokenAndRefreshToken(email);
            throw new RuntimeException("refresh token invalid : using prev version token");
        }
        return generateUserAuthResponseDto(email, UserRole.ROLE_USER);
    }
    /**
     * compare user's email and password with exist user
     * @param userSignInDto user's email and password dto
     * @return true if userSignInDto is valid
     * */
    boolean verifyUserSignInInfo(UserSignInDto userSignInDto){
        User existUser = userRepository.findByEmail(userSignInDto.getEmail()).orElse(null);
        return existUser != null && passwordEncoder.matches(userSignInDto.getPassword(), existUser.getPassword());
    }
    /**
     * save tokens on redis and return user auth response
     * @param email user's email
     * @param role user's role
    * */
    UserAuthResponseDto generateUserAuthResponseDto(String email, UserRole role){
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(email)
                .role(role)
                .build();
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        authTokenService.saveAccessTokenAndRefreshToken(email, accessToken, refreshToken);
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(email)
                .build();
    }
}
