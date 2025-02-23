package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.service.AuthManagerService;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.mail.dto.AuthCodeTemplateDto;
import com.capstone.letmedrum.mail.dto.MailDto;
import com.capstone.letmedrum.mail.service.CustomMailSenderService;
import com.capstone.letmedrum.mail.service.CustomMailTemplateService;
import com.capstone.letmedrum.auth.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.entity.UserRole;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.ZoneId;
import java.util.concurrent.ThreadLocalRandom;

@Slf4j
@Service
public class UserVerificationService {
    private final CustomMailTemplateService customMailTemplateService;
    private final CustomMailSenderService customMailSenderService;
    private final JwtUtils jwtUtils;
    private final AuthManagerService authManagerService;
    private final String senderAddress;
    /**
     * constructor for DI
     * @param customMailTemplateService CustomMailTemplateService class
     * @param customMailSenderService CustomMailSenderService class
     * @param authManagerService AuthManagerService class
     * @param jwtUtils JwtUtils class
     * @param senderAddress sender's gmail address
    * */
    public UserVerificationService(
            CustomMailTemplateService customMailTemplateService,
            CustomMailSenderService customMailSenderService,
            AuthManagerService authManagerService,
            JwtUtils jwtUtils,
            @Value("spring.name.username") String senderAddress) {
        this.customMailTemplateService = customMailTemplateService;
        this.customMailSenderService = customMailSenderService;
        this.authManagerService = authManagerService;
        this.jwtUtils = jwtUtils;
        this.senderAddress = senderAddress;
    }
    /**
     * generate email auth code
     * @param length length of auth code
     * @return email auth code
     * */
    public String generateAuthCode(int length) {
        return String.valueOf(ThreadLocalRandom.current().nextInt((int)Math.pow(10, length), (int)Math.pow(10, length+1)-1));
    }
    /**
     * send auth code to email (parameter)
     * @param email receiver's email
     * @return true if success - not null
    * */
    public boolean sendVerificationEmail(String email) {
        String authCode = generateAuthCode(5);
        AuthCodeTemplateDto templateDto = AuthCodeTemplateDto.builder()
                .title("이메일 인증번호")
                .content("아래 인증코드를 입력해주세요")
                .authCode(authCode)
                .build();
        // generate HTML page for auth code
        String template = customMailTemplateService.generateAuthCodeTemplate(templateDto);
        MailDto mailDto = MailDto.builder()
                .from(senderAddress)
                .to(email)
                .text(template)
                .build();
        // save auth code on Redis
        authManagerService.saveAuthCode(email, authCode);
        // send mail
        return customMailSenderService.sendText(mailDto, true);
    }
    /**
     * check authCode from request is same with saved authCode
     * @param email user's email
     * @param authCode authCode from request
     * @return true if authCode is same with saved authCode
    * */
    public boolean isValidAuthCode(String email, String authCode) {
        String storedAuthCode = authManagerService.getAuthCode(email);
        return authCode.equals(storedAuthCode);
    }
    /**
     * generate and save email token by user's email
     * @param email user's email
     * @return email token made by user's email
    * */
    public String createEmailVerificationToken(String email) {
        String emailToken = jwtUtils.generateJwtToken(
                new UserAuthInfoDto(email, UserRole.ROLE_USER),
                ZoneId.systemDefault(),
                Integer.toUnsignedLong(AuthManagerService.emailTokenExpSeconds)
        );
        authManagerService.saveEmailToken(email, emailToken);
        return emailToken;
    }
}
