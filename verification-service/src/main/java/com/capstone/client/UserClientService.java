package com.capstone.client;

import com.capstone.dto.UserResponseDto;
import com.capstone.exception.InvalidRequestException;
import com.capstone.response.CustomResponseDto;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
public class UserClientService {
    private final WebClient userWebClient;
    public UserClientService(@Qualifier("userWebClient") WebClient userWebClient) {
        this.userWebClient = userWebClient;
    }
    public Mono<UserResponseDto> findUserByEmail(String email) {
        return userWebClient.get()
                .uri(builder -> builder.path("/users/email")
                        .queryParam("email", email).build())
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, res -> {
                    if (res.statusCode() == HttpStatus.NOT_FOUND) return Mono.empty();
                    return Mono.error(new InvalidRequestException("Client Error"));
                })
                .bodyToMono(String.class)
                .flatMap(response -> Mono.justOrEmpty(CustomResponseDto.resolveBody(response, UserResponseDto.class)));
    }
    public Mono<UserResponseDto> findUserByNickname(String nickname) {
        return userWebClient.get()
                .uri(builder -> builder.path("/users/nickname")
                        .queryParam("nickname", nickname).build())
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, res -> {
                    if (res.statusCode() == HttpStatus.NOT_FOUND) return Mono.empty();
                    return Mono.error(new InvalidRequestException("Client Error"));
                })
                .bodyToMono(String.class)
                .flatMap(response -> Mono.justOrEmpty(CustomResponseDto.resolveBody(response, UserResponseDto.class)));
    }

}
