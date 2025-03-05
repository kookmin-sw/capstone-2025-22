package com.capstone.service;

import com.capstone.auth.UserAuthInfoDto;
import com.capstone.dto.AuthCodeTemplateDto;
import com.capstone.dto.MailDto;
import com.capstone.enums.UserRole;
import com.capstone.jwt.JwtUtils;
import com.capstone.util.VerificationInfoManager;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.time.ZoneId;
import java.util.concurrent.ThreadLocalRandom;

@Slf4j
@Service
public class VerificationService {
    @Value("spring.mail.username")
    private String senderAddress;

    private final CustomMailTemplateService customMailTemplateService;
    private final CustomMailSenderService customMailSenderService;
    private final VerificationInfoManager verificationInfoManager;
    private final JwtUtils jwtUtils;
    /**
     * constructor for DI
     * @param customMailTemplateService CustomMailTemplateService class
     * @param customMailSenderService CustomMailSenderService class
     * @param jwtUtils JwtUtils class
    * */
    public VerificationService(
            CustomMailTemplateService customMailTemplateService,
            CustomMailSenderService customMailSenderService,
            VerificationInfoManager verificationInfoManager,
            JwtUtils jwtUtils) {
        this.customMailTemplateService = customMailTemplateService;
        this.customMailSenderService = customMailSenderService;
        this.verificationInfoManager = verificationInfoManager;
        this.jwtUtils = jwtUtils;
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
     * @return Mono<Boolean> if success - not null
    * */
    public Mono<Boolean> sendVerificationEmail(String email) {
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
        return verificationInfoManager.saveAuthCode(email, authCode)
                .flatMap(res -> {
                    return Mono.just(customMailSenderService.sendText(mailDto, true));
                });
    }
    /**
     * check authCode from request is same with saved authCode
     * @param email user's email
     * @param authCode authCode from request
     * @return Mono<Boolean> if authCode is same with saved authCode
    * */
    public Mono<Boolean> isValidAuthCode(String email, String authCode) {
        return verificationInfoManager.getAuthCode(email)
                .map(res -> res.equals(authCode));
    }
    /**
     * generate and save email token by user's email
     * @param email user's email
     * @return email token made by user's email
    * */
    public Mono<String> createEmailVerificationToken(String email) {
        String emailToken = jwtUtils.generateJwtToken(
                new UserAuthInfoDto(email, UserRole.ROLE_USER),
                ZoneId.systemDefault(),
                Integer.toUnsignedLong(VerificationInfoManager.EMAIL_TOKEN_EXPIRE)
        );
        return verificationInfoManager.saveEmailToken(email, emailToken)
                .flatMap(res -> Mono.just(emailToken));
    }
}
