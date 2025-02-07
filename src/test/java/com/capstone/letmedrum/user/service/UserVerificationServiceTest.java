package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.common.service.RedisSingleDataService;
import com.capstone.letmedrum.mail.service.CustomMailSenderService;
import com.capstone.letmedrum.mail.service.CustomMailTemplateService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserVerificationServiceTest {
    @InjectMocks
    private UserVerificationService userVerificationService;
    @Mock
    CustomMailSenderService mailSenderService;
    @Mock
    CustomMailTemplateService mailTemplateService;
    @Mock
    RedisSingleDataService redisSingleDataService;

    @Test
    @DisplayName("인증메일 전송 테스트")
    void sendVerificationEmailTest() {
        // given
        String email = "test@test.com";
        // stub
        when(mailTemplateService.generateAuthCodeTemplate(any())).thenReturn("mail template");
        when(redisSingleDataService.setValue(anyString(), anyString(), anyInt())).thenReturn(1);
        when(mailSenderService.sendText(any(), anyBoolean())).thenReturn(true);
        // when
        boolean result = userVerificationService.sendVerificationEmail(email);
        // then
        assertTrue(result);
    }

    @Test
    @DisplayName("인증메일 검증 테스트")
    void isValidAuthCodeTest() {
        // given
        String email = "test@test.com";
        String authCode = "123456";
        // stub
        when(redisSingleDataService.getValue(anyString())).thenReturn(authCode);
        // when
        boolean result = userVerificationService.isValidAuthCode(email, authCode);
        // then
        assert(result);
    }
}