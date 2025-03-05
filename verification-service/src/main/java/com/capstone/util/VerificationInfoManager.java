package com.capstone.util;

import com.capstone.exception.CustomException;
import com.capstone.exception.InternalServerException;
import com.capstone.exception.InvalidRequestException;
import com.capstone.redis.RedisSingleDataServiceImpl;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

@Component
public class VerificationInfoManager {
    @Value("${verification.email-token-expire}")
    public static int EMAIL_TOKEN_EXPIRE;
    @Value("${verification.auth-code-expire}")
    public static int AUTH_CODE_EXPIRE;
    private final RedisSingleDataServiceImpl redisSingleDataService;
    public VerificationInfoManager(RedisSingleDataServiceImpl redisSingleDataService) {
        this.redisSingleDataService = redisSingleDataService;
    }
    /**
     * generate redis key with email
     * @param email users email
     * @return redis key - String, not-null
     * */
    public String getAuthCodeKey(String email) {
        String authCodeKeySuffix = "_authCode";
        return email + authCodeKeySuffix;
    }
    /**
     * generate email token key for redis
     * @param email user's email
     * @return user's email token key, not-null
     * */
    public String getEmailTokenKey(String email){
        String suffix = "_email_token";
        return email + suffix;
    }
    /**
     * save email token on redis
     * @param email user's email to generate token key
     * @param authCode user's email token for edit password
     * @throws RuntimeException if failed to save email token
     * */
    @Transactional
    public Mono<Boolean> saveAuthCode(String email, String authCode){
        String authCodeKey = getAuthCodeKey(email);
        return saveAuthInfo(authCodeKey, authCode, AUTH_CODE_EXPIRE);
    }
    /**
     * save email token on redis
     * @param email user's email to generate token key
     * @param emailToken user's email token for edit password
     * @throws RuntimeException if failed to save email token
     * */
    @Transactional
    public Mono<Boolean> saveEmailToken(String email, String emailToken){
        String emailTokenKey = getEmailTokenKey(email);
        return saveAuthInfo(emailTokenKey, emailToken, EMAIL_TOKEN_EXPIRE);
    }
    /**
     * save auth info on redis
     * @param key key of data to save
     * @param value value of data to save
     * @param expSeconds exp time of data (unit : second)
     * @throws RuntimeException if failed to save data
     * */
    @Transactional
    public Mono<Boolean> saveAuthInfo(String key, String value, int expSeconds){
        return redisSingleDataService.setValue(key, value, expSeconds)
                .flatMap(res -> {
                   if(!res) return Mono.error(new InternalServerException("failed to save verification info"));
                   return Mono.just(true);
                });
    }
    /**
     * get auth code from redis
     * @param email user's email to find auth code
     * @return auth code - String
     * */
    public Mono<String> getAuthCode(String email){
        String authCodeKey = getAuthCodeKey(email);
        return redisSingleDataService.getValue(authCodeKey)
                .switchIfEmpty(Mono.error(new InvalidRequestException("failed to get verification code")));
    }
    /**
     * get email token from redis
     * @param email user's email to find email token
     * @return email token - String
     * */
    public Mono<String> getEmailToken(String email){
        String emailTokenKey = getEmailTokenKey(email);
        return redisSingleDataService.getValue(emailTokenKey);
    }
}
