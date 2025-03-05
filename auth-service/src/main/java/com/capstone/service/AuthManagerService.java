package com.capstone.service;

import com.capstone.exception.InternalServerException;
import com.capstone.jwt.JwtUtils;
import com.capstone.redis.RedisSingleDataServiceImpl;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

@Service
public class AuthManagerService {
    private final RedisSingleDataServiceImpl redisSingleDataServiceImpl;
    public AuthManagerService(RedisSingleDataServiceImpl redisSingleDataServiceImpl) {
        this.redisSingleDataServiceImpl = redisSingleDataServiceImpl;
    }
    /**
     * generate access token key for redis
     * @param email user's email
     * @return user's access token key, not-null
     * */
    public String getAccessTokenKey(String email){
        String suffix = "_access_token";
        return email + suffix;
    }
    /**
     * generate refresh token key for redis
     * @param email user's email
     * @return user's refresh token key, not-null
     * */
    public String getRefreshTokenKey(String email){
        String suffix = "_refresh_token";
        return email + suffix;
    }
    /**
     * save access token and refresh token on redis
     * @param email user's email to generate token key
     * @param accessToken user's access token
     * @param refreshToken user's refresh token
     * @throws RuntimeException if failed to save access token and refresh token
     * */
    @Transactional
    public void saveAccessTokenAndRefreshToken(String email, String accessToken, String refreshToken){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        saveAuthInfo(accessTokenKey, accessToken, JwtUtils.ACCESS_TOKEN_EXP_TIME.intValue()).subscribe();
        saveAuthInfo(refreshTokenKey, refreshToken, JwtUtils.REFRESH_TOKEN_EXP_TIME.intValue()).subscribe();
    }
    /**
     * save auth info on redis
     * @param key key of data to save
     * @param value value of data to save
     * @param expSeconds exp time of data (unit : second)
     * @throws RuntimeException if failed to save data
    * */
    @Transactional
    public Mono<Void> saveAuthInfo(String key, String value, int expSeconds){
        return redisSingleDataServiceImpl.setValue(key, value, expSeconds)
                .flatMap( res -> res ? Mono.empty() : Mono.error(new InternalServerException("Error saving auth info")) );
    }
    /**
     * delete access token and refresh token on redis
     * @param email user's email to delete token
     * @throws RuntimeException if failed to delete token
     * */
    @Transactional
    public Mono<Boolean> deleteAccessTokenAndRefreshToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        return Mono.zip(redisSingleDataServiceImpl.deleteValue(accessTokenKey), redisSingleDataServiceImpl.deleteValue(refreshTokenKey))
                .flatMap(res -> {
                    if(!res.getT1() || !res.getT2()) return Mono.error(new InternalServerException("Error delete auth info"));
                    return Mono.just(true);
                });
    }
    /**
     * get access token from redis
     * @param email user's email to find access token
     * @return access token - String
    * */
    public Mono<String> getAccessToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        return redisSingleDataServiceImpl.getValue(accessTokenKey);
    }
    /**
     * get refresh token from redis
     * @param email user's email to find refresh token
     * @return refresh token - String
     * */
    public Mono<String> getRefreshToken(String email){
        String refreshTokenKey = getRefreshTokenKey(email);
        return redisSingleDataServiceImpl.getValue(refreshTokenKey);
    }
}
