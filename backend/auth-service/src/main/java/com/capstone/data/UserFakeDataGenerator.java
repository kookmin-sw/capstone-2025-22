package com.capstone.data;

import com.capstone.dto.request.SignUpDto;
import com.capstone.service.UserAuthService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@Profile("dev")
@RequiredArgsConstructor
public class UserFakeDataGenerator {
    private final UserAuthService userAuthService;

    @PostConstruct
    public void init() {
        userAuthService.signUpUser(SignUpDto.builder()
                .email("test@test.com")
                .nickname("testUser")
                .password("1234").build()).subscribe();
    }
}
