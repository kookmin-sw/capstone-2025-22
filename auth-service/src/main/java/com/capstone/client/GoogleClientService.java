package com.capstone.client;

import com.capstone.dto.google.GoogleTokenRequestDto;
import com.capstone.dto.google.GoogleTokenResponseDto;
import com.capstone.dto.google.GoogleUserInfoResponseDto;
import com.capstone.exception.InternalServerException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;

@Service
public class GoogleClientService {
    @Value("${auth.google.client-id}")
    private String clientId;
    @Value("${auth.google.client-secret}")
    private String clientSecret;
    @Value("${auth.google.redirect-uri}")
    private String redirectUri;
    @Value("${auth.google.access-token-uri}")
    private String googleAccessTokenUri;
    @Value("${auth.google.user-info-uri}")
    private String googleUserInfoUri;
    private final WebClient webClient;
    public GoogleClientService(WebClient.Builder webClientBuilder) {
        webClient = webClientBuilder.build();
    }
    /**
     * get Google access token by using authentication code
     * @param authCode client's authentication code
     * @return access token from Google
     * @throws InternalServerException if failed to retrieve access token
     * */
    public Mono<String> getAccessToken(String authCode) {
        String decodedAuthCode = URLDecoder.decode(authCode, StandardCharsets.UTF_8);
        GoogleTokenRequestDto googleTokenRequestDto = GoogleTokenRequestDto.builder()
                .client_id(clientId)
                .client_secret(clientSecret)
                .redirect_uri(redirectUri)
                .code(decodedAuthCode)
                .grant_type("authorization_code")
                .build();
        return webClient.post()
                .uri(googleAccessTokenUri)
                .accept(MediaType.APPLICATION_JSON)
                .bodyValue(googleTokenRequestDto)
                .retrieve()
                .onStatus(HttpStatusCode::isError, res -> Mono.error(new InternalServerException("failed to get access token")))
                .bodyToMono(GoogleTokenResponseDto.class)
                .retry(2)
                .map(GoogleTokenResponseDto::getAccess_token);
    }
    /**
     * get user info from Google by using access token
     * @param accessToken user's access token from Google
     * @return user's info from Google
     * @throws InternalServerException if failed to retrieve info
     * */
    public Mono<GoogleUserInfoResponseDto> getUserInfo(String accessToken) {
        return webClient.get()
                .uri(googleUserInfoUri)
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .retrieve()
                .onStatus(HttpStatusCode::isError, res -> Mono.error(new InternalServerException("failed to load user info")))
                .bodyToMono(GoogleUserInfoResponseDto.class)
                .retry(2);
    }
}
