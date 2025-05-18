package com.capstone.jwt;

import com.capstone.auth.UserAuthInfoDto;
import com.capstone.enums.UserRole;
import org.junit.jupiter.api.Test;

import java.time.ZoneId;
import static java.time.ZoneId.SHORT_IDS;

import static org.junit.jupiter.api.Assertions.*;


class JwtUtilsTest {

    String testSecrets = "testSecretstestSecretstestSecretstestSecretstestSecretstestSecretstestSecretstestSecretstestSecretstestSecrets";
    JwtUtils jwtUtils = new JwtUtils(testSecrets);

    @Test
    void verification_test(){
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto.builder()
                .email("asdf@test.com")
                .role(UserRole.ROLE_USER)
                .build();

        String commonToken = jwtUtils.generateJwtToken(userAuthInfoDto, ZoneId.of(SHORT_IDS.get("PNT")), 10000L);
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        String refreshToken = jwtUtils.generateRefreshToken(userAuthInfoDto);

        assert jwtUtils.validateToken(commonToken);
        assert jwtUtils.validateToken(accessToken);
        assert jwtUtils.validateToken(refreshToken);
    }
}