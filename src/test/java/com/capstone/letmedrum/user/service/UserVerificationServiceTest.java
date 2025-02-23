package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.service.AuthManagerService;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.mail.dto.MailDto;
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
    AuthManagerService authManagerService;
    @Mock
    JwtUtils jwtUtils;
    @Test
    @DisplayName("인증메일 전송 성공 테스트")
    void sendVerificationEmailTest() {
        // given
        String email = "test@test.com";
        String emailTemplate = "emailTemplateHtml";
        // stub
        when(mailTemplateService.generateAuthCodeTemplate(any())).thenReturn(emailTemplate);
        when(mailSenderService.sendText(any(MailDto.class), anyBoolean())).thenReturn(true);
        // when
        boolean result = userVerificationService.sendVerificationEmail(email);
        // then
        assertTrue(result);
    }
    @Test
    @DisplayName("인증메일 검증 성공/실패 테스트")
    void isValidAuthCodeTest() {
        // given
        String email = "test@test.com";
        String wrongEmail = "wrong@test.com";
        String authCode = "123456";
        // stub
        when(authManagerService.getAuthCode(email)).thenReturn(authCode);
        when(authManagerService.getAuthCode(wrongEmail)).thenReturn("wrongAuthCode");
        // when
        boolean expectedTrue = userVerificationService.isValidAuthCode(email, authCode);
        boolean expectedFalse = userVerificationService.isValidAuthCode(wrongEmail, authCode);
        // then
        assertTrue(expectedTrue);
        assertFalse(expectedFalse);
    }
    @Test
    @DisplayName("이메일 인증 성공 토큰 생성")
    void createEmailVerificationTokenTest(){
        // given
        String emailToken = "test.test.com";
        String email = "test@test.com";
        // stub
        when(jwtUtils.generateJwtToken(any(), any(), any())).thenReturn(emailToken);
        // when
        String createdEmailToken = userVerificationService.createEmailVerificationToken(email);
        // then
        assertNotNull(createdEmailToken);
        assertEquals(emailToken, createdEmailToken);
    }
}