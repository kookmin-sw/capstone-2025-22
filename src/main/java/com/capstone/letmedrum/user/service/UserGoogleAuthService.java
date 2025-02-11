package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.user.dto.*;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriUtils;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

@Service
public class UserGoogleAuthService {
    @Value("${env.auth.google.client_id}")
    private String clientId;
    @Value("${env.auth.google.client_secret}")
    private String clientSecret;
    @Value("${env.auth.google.redirect_uri}")
    private String redirectUri;
    private final RestTemplate restTemplate = new RestTemplate();
    private final UserRepository userRepository;
    private final UserAuthService userAuthService;
    public UserGoogleAuthService(
            UserRepository userRepository,
            UserAuthService userAuthService
    ) {
        this.userRepository = userRepository;
        this.userAuthService = userAuthService;
    }
    /**
     * get Google access token by using authentication code
     * @param authCode client's authentication code
     * @return access token from Google
    * */
    private String getGoogleAccessToken(String authCode){
        String googleAccessTokenUri = "https://oauth2.googleapis.com/token";
        String decodedAuthCode = URLDecoder.decode(authCode, StandardCharsets.UTF_8);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE);
        HttpEntity<GoogleTokenRequestDto> httpEntity = new HttpEntity<>(
                new GoogleTokenRequestDto(clientId, clientSecret, redirectUri, decodedAuthCode, "authorization_code")
                , headers
        );
        GoogleTokenResponseDto googleTokenResponseDto = restTemplate.exchange(
                googleAccessTokenUri, HttpMethod.POST, httpEntity, GoogleTokenResponseDto.class
        ).getBody();
        return Optional.ofNullable(googleTokenResponseDto)
                .orElseThrow(() -> new RuntimeException("")).getAccess_token();
    }
    /**
     * get user info from Google by using access token
     * @param accessToken user's access token from Google
     * @return user's info from Google
    * */
    private GoogleUserInfoResponseDto getGoogleUserInfo(String accessToken){
        String googleUserInfoUri = "https://www.googleapis.com/userinfo/v2/me";
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken);
        HttpEntity<String> httpEntity = new HttpEntity<>(headers);
        return restTemplate.exchange(
                googleUserInfoUri, HttpMethod.GET, httpEntity, GoogleUserInfoResponseDto.class
        ).getBody();
    }
    /**
     * sign in user by using Google auth code
     * @param authCode authentication Code from Google
     * @return UserAuthResponseDto with access token and refresh token
    * */
    @Transactional
    public UserAuthResponseDto signIn(String authCode){
        String accessToken = getGoogleAccessToken(authCode);
        GoogleUserInfoResponseDto userInfo = getGoogleUserInfo(accessToken);
        User existUser = userRepository.findByEmail(userInfo.getEmail()).orElse(null);
        if(existUser == null){
            userRepository.save(
                    User.builder()
                            .email(userInfo.getEmail())
                            .role(UserRole.ROLE_USER)
                            .password(null)
                            .nickname(userInfo.getName()).build()
            );
        }
        return userAuthService.generateUserAuthResponseDto(userInfo.getEmail(), UserRole.ROLE_USER);
    }
}
