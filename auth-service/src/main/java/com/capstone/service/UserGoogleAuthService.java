package com.capstone.service;

import com.capstone.client.GoogleClientService;
import com.capstone.client.UserClientService;
import com.capstone.dto.UserResponseDto;
import com.capstone.dto.request.SignUpDto;
import com.capstone.dto.response.AuthResponseDto;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

import java.util.UUID;

@Service
public class UserGoogleAuthService {
    private final UserAuthService userAuthService;
    private final UserClientService userClientService;
    private final GoogleClientService googleClientService;
    public UserGoogleAuthService(
            UserAuthService userAuthService,
            UserClientService userClientService,
            GoogleClientService googleClientService
    ) {
        this.userAuthService = userAuthService;
        this.userClientService = userClientService;
        this.googleClientService = googleClientService;
    }
    /**
     * sign in user by using Google auth code
     * @param authCode authentication Code from Google
     * @return UserAuthResponseDto with access token and refresh token
    * */
    @Transactional
    public Mono<AuthResponseDto> signIn(String authCode) {
        return googleClientService.getAccessToken(authCode)
                .flatMap(googleClientService::getUserInfo)
                .flatMap(userInfo -> userClientService
                        .findUserByEmail(userInfo.getEmail())
                        .switchIfEmpty(userClientService.saveUser(SignUpDto.builder()
                                .email(userInfo.getEmail())
                                .nickname("user@"+ UUID.randomUUID()).build())))
                .map(user -> userAuthService
                        .generateResponseAndSaveToken(user.getEmail(), user.getNickname(), user.getRole()));
    }
}
