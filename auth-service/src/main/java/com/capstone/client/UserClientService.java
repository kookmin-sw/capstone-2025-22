package com.capstone.client;

import com.capstone.dto.UserResponseDto;
import com.capstone.dto.request.SignUpDto;
import com.capstone.exception.InternalServerException;
import com.capstone.response.CustomResponseDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.UUID;

@Slf4j
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
                .exchangeToMono(res -> {
                    if(res.statusCode().is4xxClientError()) return Mono.empty();
                    else if(res.statusCode().is5xxServerError()) return Mono.error(new InternalServerException("Server Error : findUserByEmail"));
                    return res.bodyToMono(String.class).map(resBody->CustomResponseDto.resolveBody(resBody, UserResponseDto.class));
                });
    }
    public Mono<UserResponseDto> saveUser(SignUpDto user) {
        return userWebClient.post()
                .uri("/users")
                .bodyValue(user)
                .retrieve()
                .onStatus(HttpStatusCode::isError, res -> Mono.error(new InternalServerException("failed to save user")))
                .bodyToMono(String.class)
                .flatMap( response -> Mono.just(CustomResponseDto.resolveBody(response, UserResponseDto.class)));
    }
    public Mono<UserResponseDto> saveUser(UserResponseDto user) {
        SignUpDto newUser = SignUpDto.builder()
                .email(user.getEmail())
                .nickname("user@" + UUID.randomUUID())
                .build();
        return saveUser(newUser);
    }
}
