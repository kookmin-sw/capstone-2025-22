package com.capstone.letmedrum.auth.service;

import com.capstone.letmedrum.auth.dto.GoogleTokenRequestDto;
import com.capstone.letmedrum.auth.dto.GoogleTokenResponseDto;
import com.capstone.letmedrum.auth.dto.GoogleUserInfoResponseDto;
import com.capstone.letmedrum.common.exception.CustomException;
import com.capstone.letmedrum.common.exception.InvalidRequestException;
import com.capstone.letmedrum.auth.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.UUID;

@Service
public class UserGoogleAuthService {
    @Value("${env.auth.google.client-id}")
    private String clientId;
    @Value("${env.auth.google.client-secret}")
    private String clientSecret;
    @Value("${env.auth.google.redirect-uri}")
    private String redirectUri;
    @Value("${env.auth.google.access-token-uri}")
    private String googleAccessTokenUri;
    @Value("${env.auth.google.user-info-uri}")
    private String googleUserInfoUri;
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
     * @throws InvalidRequestException if failed to retrieve access token
    * */
    public String getGoogleAccessToken(String authCode){
        String decodedAuthCode = URLDecoder.decode(authCode, StandardCharsets.UTF_8);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE);
        HttpEntity<GoogleTokenRequestDto> httpEntity = new HttpEntity<>(
                new GoogleTokenRequestDto(clientId, clientSecret, redirectUri, decodedAuthCode, "authorization_code")
                , headers
        );
        try{
            GoogleTokenResponseDto googleTokenResponseDto = restTemplate.exchange(
                    googleAccessTokenUri, HttpMethod.POST, httpEntity, GoogleTokenResponseDto.class
            ).getBody();
            return Optional.ofNullable(googleTokenResponseDto)
                    .orElseThrow(() -> new RuntimeException("")).getAccess_token();
        }catch (RestClientException e){
            throw new InvalidRequestException(e.getMessage());
        }
    }
    /**
     * get user info from Google by using access token
     * @param accessToken user's access token from Google
     * @return user's info from Google
     * @throws CustomException if failed to retrieve info
    * */
    public GoogleUserInfoResponseDto getGoogleUserInfo(String accessToken){
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken);
        HttpEntity<String> httpEntity = new HttpEntity<>(headers);
        try {
            return restTemplate.exchange(
                    googleUserInfoUri, HttpMethod.GET, httpEntity, GoogleUserInfoResponseDto.class
            ).getBody();
        }catch (RestClientException e){
            throw new CustomException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
        }
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
                            .nickname("user@" + UUID.randomUUID()).build()
            );
        }
        return userAuthService.generateResponseAndSaveToken(userInfo.getEmail(), UserRole.ROLE_USER);
    }
}
