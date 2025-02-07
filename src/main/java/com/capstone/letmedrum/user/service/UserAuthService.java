package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.common.service.RedisSingleDataService;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.dto.UserSignInDto;
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
    private final RedisSingleDataService redisSingleDataService;
    public UserAuthService(
            UserRepository userRepository,
            JwtUtils jwtUtils,
            PasswordEncoder passwordEncoder,
            UserRetrieveService userRetrieveService,
            RedisSingleDataService redisSingleDataService
    ) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
        this.redisSingleDataService = redisSingleDataService;
    }
    String getAccessTokenKey(String email){
        String suffix = "_access_token";
        return email + suffix;
    }
    String getRefreshTokenKey(String email){
        String suffix = "_refresh_token";
        return email + suffix;
    }
    void saveAccessTokenAndRefreshToken(String email, String accessToken, String refreshToken){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        if(redisSingleDataService.setValue(accessTokenKey, accessToken, JwtUtils.ACCESS_TOKEN_EXP_TIME.intValue())==0 ||
                redisSingleDataService.setValue(refreshTokenKey, refreshToken, JwtUtils.REFRESH_TOKEN_EXP_TIME.intValue())==0
        ){
            throw new RuntimeException("failed to set tokens");
        }
    }
    void deleteAccessTokenAndRefreshToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        if(redisSingleDataService.deleteValue(accessTokenKey)==0 || redisSingleDataService.deleteValue(refreshTokenKey)==0){
            throw new RuntimeException("failed to delete tokens");
        }
    }
    /**
     * Check user's id and password and return auth info
     * @param userSignInDto UserSignInDto, not-null
     * @return UserAuthResponseDto - not null
     * @throws RuntimeException If user info is not valid
    * */
    public UserAuthResponseDto signInUser(UserSignInDto userSignInDto) {
        User existUser = userRepository.findByEmail(userSignInDto.getEmail()).orElse(null);
        if(existUser==null || !passwordEncoder.matches(userSignInDto.getPassword(), existUser.getPassword())){
            throw new RuntimeException("sign in error : invalid email or password");
        }
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(existUser.getEmail())
                .role(existUser.getRole())
                .build();
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        saveAccessTokenAndRefreshToken(existUser.getEmail(), accessToken, refreshToken);
        return UserAuthResponseDto
                .builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
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
                .role(user.getRole())
                .build();
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        saveAccessTokenAndRefreshToken(user.getEmail(), accessToken, refreshToken);
        return UserAuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(user.getEmail())
                .build();
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
        deleteAccessTokenAndRefreshToken(email);
        return true;
    }
    /**
     * regenerate access token and refresh token
     * @param refreshToken - user's refresh token, String
     * @return UserAuthResponseDto - UserAuthResponseDto, not-null
     * @throws RuntimeException - if prev version or invalid token is sent
    * */
    @Transactional
    public UserAuthResponseDto doRefreshTokenRotation(String refreshToken){
        if(!jwtUtils.validateToken(refreshToken)){
            throw new RuntimeException("refresh token invalid : using invalid or expired token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        String savedRefreshToken = redisSingleDataService.getValue(getRefreshTokenKey(email));
        if(!savedRefreshToken.equals(refreshToken)){
            deleteAccessTokenAndRefreshToken(email);
            throw new RuntimeException("refresh token invalid : using prev version token");
        }
        UserAuthInfoDto userAuthInfoDto = new UserAuthInfoDto(
                userRepository.findByEmail(email).orElseThrow(() -> new RuntimeException("invalid email"))
        );
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String newRefreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        saveAccessTokenAndRefreshToken(email, accessToken, newRefreshToken);
        return UserAuthResponseDto.builder()
                .email(email)
                .accessToken(accessToken)
                .refreshToken(newRefreshToken).build();
    }
}
