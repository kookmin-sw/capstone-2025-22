package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.common.service.RedisSingleDataService;
import com.capstone.letmedrum.mail.dto.AuthCodeTemplateDto;
import com.capstone.letmedrum.mail.dto.MailDto;
import com.capstone.letmedrum.mail.service.CustomMailSenderService;
import com.capstone.letmedrum.mail.service.CustomMailTemplateService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.concurrent.ThreadLocalRandom;

@Slf4j
@Service
public class UserVerificationService {
    private final CustomMailTemplateService customMailTemplateService;
    private final CustomMailSenderService customMailSenderService;
    private final RedisSingleDataService redisSingleDataService;
    private final String senderAddress;
    /**
     * constructor for DI
     * @param customMailTemplateService CustomMailTemplateService class
     * @param customMailSenderService CustomMailSenderService class
     * @param redisSingleDataService RedisSingleDataService class
     * @param senderAddress sender's gmail address
    * */
    public UserVerificationService(
            CustomMailTemplateService customMailTemplateService,
            CustomMailSenderService customMailSenderService,
            RedisSingleDataService redisSingleDataService,
            @Value("spring.name.username") String senderAddress) {
        this.customMailTemplateService = customMailTemplateService;
        this.customMailSenderService = customMailSenderService;
        this.redisSingleDataService = redisSingleDataService;
        this.senderAddress = senderAddress;
    }
    /**
     * generate redis key with email
     * @param email users email
     * @return redis key - String, not-null
     * */
    public String getRedisAuthCodeKey(String email) {
        String authCodeKeySuffix = "_authCode";
        return email + authCodeKeySuffix;
    }
    public String generateAuthCode(int length) {
        return String.valueOf(ThreadLocalRandom.current().nextInt((int)Math.pow(10, length), (int)Math.pow(10, length+1)-1));
    }
    /**
     * send auth code to email (parameter)
     * @param email receiver's email
     * @return true if success - not null
     * @throws RuntimeException if failed to save auth code on redis
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
        if(redisSingleDataService.setValue(getRedisAuthCodeKey(email), authCode, 600)==0){
            log.error("failed to save authCode on Redis");
            throw new RuntimeException("failed to save authCode on Redis");
        }
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
        String redisAuthCodeKey = getRedisAuthCodeKey(email);
        String storedAuthCode = redisSingleDataService.getValue(redisAuthCodeKey);
        return authCode.equals(storedAuthCode);
    }
}
