package com.capstone.letmedrum.auth.service;

import com.capstone.letmedrum.common.exception.CustomException;
import com.capstone.letmedrum.common.exception.DataNotFoundException;
import com.capstone.letmedrum.common.service.RedisSingleDataServiceImpl;
import com.capstone.letmedrum.config.security.JwtUtils;
import jakarta.transaction.Transactional;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class AuthManagerService {
    public static final int authCodeExpSeconds = 600;
    public static final int emailTokenExpSeconds = 10000;
    private final RedisSingleDataServiceImpl redisSingleDataServiceImpl;
    public AuthManagerService(RedisSingleDataServiceImpl redisSingleDataServiceImpl) {
        this.redisSingleDataServiceImpl = redisSingleDataServiceImpl;
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
     * generate email token key for redis
     * @param email user's email
     * @return user's email token key, not-null
     * */
    public String getEmailTokenKey(String email){
        String suffix = "_email_token";
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
        saveAuthInfo(accessTokenKey, accessToken, JwtUtils.ACCESS_TOKEN_EXP_TIME.intValue());
        saveAuthInfo(refreshTokenKey, refreshToken, JwtUtils.REFRESH_TOKEN_EXP_TIME.intValue());
    }
    /**
     * save email token on redis
     * @param email user's email to generate token key
     * @param authCode user's email token for edit password
     * @throws RuntimeException if failed to save email token
     * */
    @Transactional
    public void saveAuthCode(String email, String authCode){
        String authCodeKey = getAuthCodeKey(email);
        saveAuthInfo(authCodeKey, authCode, authCodeExpSeconds);
    }
    /**
     * save email token on redis
     * @param email user's email to generate token key
     * @param emailToken user's email token for edit password
     * @throws RuntimeException if failed to save email token
     * */
    @Transactional
    public void saveEmailToken(String email, String emailToken){
        String emailTokenKey = getEmailTokenKey(email);
        saveAuthInfo(emailTokenKey, emailToken, emailTokenExpSeconds);
    }
    /**
     * save auth info on redis
     * @param key key of data to save
     * @param value value of data to save
     * @param expSeconds exp time of data (unit : second)
     * @throws RuntimeException if failed to save data
    * */
    @Transactional
    public void saveAuthInfo(String key, String value, int expSeconds){
        if(!redisSingleDataServiceImpl.setValue(key, value, expSeconds)){
            throw new CustomException(HttpStatus.INTERNAL_SERVER_ERROR, "failed to save auth info");
        }
    }
    /**
     * delete access token and refresh token on redis
     * @param email user's email to delete token
     * @throws RuntimeException if failed to delete token
     * */
    @Transactional
    public void deleteAccessTokenAndRefreshToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        String refreshTokenKey = getRefreshTokenKey(email);
        if(!redisSingleDataServiceImpl.deleteValue(accessTokenKey) || !redisSingleDataServiceImpl.deleteValue(refreshTokenKey)){
            throw new CustomException(HttpStatus.INTERNAL_SERVER_ERROR, "failed to delete auth info");
        }
    }
    /**
     * get access token from redis
     * @param email user's email to find access token
     * @return access token - String
    * */
    public String getAccessToken(String email){
        String accessTokenKey = getAccessTokenKey(email);
        return redisSingleDataServiceImpl.getValue(accessTokenKey);
    }
    /**
     * get refresh token from redis
     * @param email user's email to find refresh token
     * @return refresh token - String
     * */
    public String getRefreshToken(String email){
        String refreshTokenKey = getRefreshTokenKey(email);
        return redisSingleDataServiceImpl.getValue(refreshTokenKey);
    }
    /**
     * get auth code from redis
     * @param email user's email to find auth code
     * @return auth code - String
     * */
    public String getAuthCode(String email){
        String authCodeKey = getAuthCodeKey(email);
        return redisSingleDataServiceImpl.getValue(authCodeKey);
    }
    /**
     * get email token from redis
     * @param email user's email to find email token
     * @return email token - String
     * */
    public String getEmailToken(String email){
        String emailTokenKey = getEmailTokenKey(email);
        return redisSingleDataServiceImpl.getValue(emailTokenKey);
    }
}
