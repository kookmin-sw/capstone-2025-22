package com.capstone.service;

import com.capstone.client.UserClientService;
import com.capstone.dto.UserResponseDto;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
public class UserInfoVerificationService {
    private final UserClientService userClientService;
    public UserInfoVerificationService(UserClientService userClientService) {
        this.userClientService = userClientService;
    }
    public Mono<Boolean> isValidEmail(String email) {
        return userClientService.findUserByEmail(email)
                .flatMap(user -> Mono.just(false))
                .switchIfEmpty(Mono.just(true));
    }
    public Mono<Boolean> isValidNickname(String nickname) {
        return userClientService.findUserByNickname(nickname)
                .flatMap(user -> Mono.just(false))
                .switchIfEmpty(Mono.just(true));
    }
}
