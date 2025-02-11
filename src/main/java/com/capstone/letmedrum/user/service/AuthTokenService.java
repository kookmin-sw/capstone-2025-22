package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.common.service.RedisSingleDataServiceImpl;
import com.capstone.letmedrum.config.security.JwtUtils;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

@Service
public class AuthTokenService {
    private final RedisSingleDataServiceImpl redisSingleDataServiceImpl;
    public AuthTokenService(RedisSingleDataServiceImpl redisSingleDataServiceImpl) {
        this.redisSingleDataServiceImpl = redisSingleDataServiceImpl;
    }
    /**
     * generate access token key for redis
     * @param email user's email
     * @return user's access token key, not-null
     * */
    String getAccessTokenKey(String email){
        String suffix = "_access_token";
        return email + suffix;
    }
    /**
     * generate refresh token key for redis
     * @param email user's email
     * @return user's refresh token key, not-null
     * */
    String getRefreshTokenKey(String email){
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
    void saveAccessTokenAndRefreshToken(String email, String accessToken, String refreshToken){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        if(!redisSingleDataServiceImpl.setValue(accessTokenKey, accessToken, JwtUtils.ACCESS_TOKEN_EXP_TIME.intValue()) ||
                !redisSingleDataServiceImpl.setValue(refreshTokenKey, refreshToken, JwtUtils.REFRESH_TOKEN_EXP_TIME.intValue())
        ){
            throw new RuntimeException("failed to set tokens");
        }
    }
    /**
     * delete access token and refresh token on redis
     * @param email user's email to delete token
     * @throws RuntimeException if failed to delete token
     * */
    @Transactional
    void deleteAccessTokenAndRefreshToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        if(!redisSingleDataServiceImpl.deleteValue(accessTokenKey) || !redisSingleDataServiceImpl.deleteValue(refreshTokenKey)){
            throw new RuntimeException("failed to delete tokens");
        }
    }
    /**
     * get access token from redis
     * @param email user's email to find access token
     * @return access token - String
    * */
    String getAccessToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        return redisSingleDataServiceImpl.getValue(accessTokenKey);
    }
    /**
     * get refresh token from redis
     * @param email user's email to find refresh token
     * @return refresh token - String
     * */
    String getRefreshToken(String email){
        String refreshTokenKey = getRefreshTokenKey(email);
        return redisSingleDataServiceImpl.getValue(refreshTokenKey);
    }
}
