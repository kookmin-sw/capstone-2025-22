package com.capstone.service;

import com.capstone.auth.UserAuthInfoDto;
import com.capstone.client.UserClientService;
import com.capstone.dto.UserResponseDto;
import com.capstone.dto.response.AuthResponseDto;
import com.capstone.dto.request.SignInDto;
import com.capstone.dto.request.SignUpDto;
import com.capstone.enums.UserRole;
import com.capstone.exception.CustomException;
import com.capstone.exception.InvalidTokenException;
import com.capstone.exception.InvalidUserInfoException;
import com.capstone.jwt.JwtUtils;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

@Service
public class UserAuthService {
    private final JwtUtils jwtUtils;
    private final PasswordEncoder passwordEncoder;
    private final AuthManagerService authManagerService;
    private final UserClientService userClientService;
    public UserAuthService(
            JwtUtils jwtUtils,
            PasswordEncoder passwordEncoder,
            AuthManagerService authManagerService,
            UserClientService userClientService
    ) {
        this.jwtUtils = jwtUtils;
        this.passwordEncoder = passwordEncoder;
        this.authManagerService = authManagerService;
        this.userClientService = userClientService;
    }
    /**
     * Check user's id and password and return auth info
     * @param signInDto UserSignInDto, not-null
     * @return UserAuthResponseDto - not null
     * @throws InvalidUserInfoException If user info is not valid
    * */
    public Mono<AuthResponseDto> signInUser(SignInDto signInDto) {
        return userClientService.findUserByEmail(signInDto.getEmail())
                .switchIfEmpty(Mono.just(new UserResponseDto()))
                .flatMap(user -> {
                    if(user.getEmail()==null || !user.getEmail().equals(signInDto.getEmail()) || !passwordEncoder.matches(signInDto.getPassword(), user.getPassword())){
                        return Mono.error(new InvalidUserInfoException("invalid email or password"));
                    }
                    return Mono.just(generateResponseAndSaveToken(user.getEmail(), user.getNickname(), user.getRole()));
                });
    }
    /**
     * Sign up the user if a user with that email does not exist.
     * @param signUpDto userSignUpDto not-null
     * @return UserAuthResponseDto - not null
     * @throws CustomException If user already exists
    * */
    @Transactional
    public Mono<AuthResponseDto> signUpUser(SignUpDto signUpDto) {
        return userClientService.findUserByEmail(signUpDto.getEmail())
                .switchIfEmpty(Mono.just(new UserResponseDto()))
                .flatMap(user -> {
                    if(user.getEmail()!=null) return Mono.error(new CustomException(HttpStatus.CONFLICT,"user already exists"));
                    signUpDto.setPassword(passwordEncoder.encode(signUpDto.getPassword()));
                    return userClientService.saveUser(signUpDto)
                            .map(res -> generateResponseAndSaveToken(signUpDto.getEmail(), signUpDto.getNickname(),UserRole.ROLE_USER));
                });
    }

    /**
     * sign out user by deleting token
     * @param refreshToken - user's refresh token
     * @return isSuccess - boolean, nullable
     * @throws InvalidTokenException if token is invalid
     * @throws InvalidUserInfoException if user not exists
    * */
    @Transactional
    public Mono<Boolean> signOutUser(String refreshToken) {
        if(!jwtUtils.validateToken(refreshToken)){
            throw new InvalidTokenException("invalid refresh token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        return authManagerService.deleteAccessTokenAndRefreshToken(email);
    }
    /**
     * regenerate access token and refresh token
     * @param refreshToken - user's refresh token, String
     * @return UserAuthResponseDto - UserAuthResponseDto, not-null
     * @throws InvalidTokenException if prev version token or invalid token is sent
    * */
    @Transactional
    public Mono<AuthResponseDto> doRefreshTokenRotation(String refreshToken){
        if(!jwtUtils.validateToken(refreshToken)){
            throw new InvalidTokenException("invalid refresh token : invalid token");
        }
        String email = jwtUtils.getUserEmail(refreshToken);
        return authManagerService.getRefreshToken(email)
                .flatMap(savedRefreshToken -> {
                    if(!savedRefreshToken.equals(refreshToken)){
                        return authManagerService.deleteAccessTokenAndRefreshToken(email)
                                .then(Mono.error(new InvalidTokenException("invalid refresh token : using prev token")));
                    }else{
                        return userClientService.findUserByEmail(email);
                    }
                })
                .flatMap(user -> Mono.just(generateResponseAndSaveToken(user.getEmail(), user.getNickname(), user.getRole())));
    }
    /**
     * save tokens on redis and return user auth response
     * @param email user's email
     * @param role user's role
    * */
    public AuthResponseDto generateResponseAndSaveToken(String email, String nickname, UserRole role){
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(email)
                .role(role)
                .build();
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        authManagerService.saveAccessTokenAndRefreshToken(email, accessToken, refreshToken);
        return AuthResponseDto
                .builder()
                .accessToken(jwtUtils.generateAccessToken(userAuthInfoDto))
                .refreshToken(jwtUtils.generateRefreshToken(userAuthInfoDto))
                .email(email)
                .nickname(nickname)
                .build();
    }
}
