package com.capstone.letmedrum.config.security;

import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.entity.UserRole;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.time.ZoneId;

@SpringBootTest
class JwtUtilsTest {
    @Autowired
    private JwtUtils jwtUtils;
    private static UserAuthInfoDto userAuthInfoDto;
    @BeforeAll
    public static void init(){
        userAuthInfoDto = UserAuthInfoDto.builder()
                .email("email")
                .role(UserRole.ROLE_ADMIN)
                .build();
    }
    @Test
    public void testGenerateJwtToken(){
        //given
        ZoneId zoneId = ZoneId.of("Asia/Seoul");
        long exp_time = 10000L;
        //when
        String token = jwtUtils.generateJwtToken(userAuthInfoDto, zoneId, exp_time);
        //then
        assert (jwtUtils.validateToken(token));
        assert (jwtUtils.getUserEmail(token).equals("email"));
    }

    @Test
    public void testGenerateAccessToken(){
        //given
        //when
        String accessToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        //then
        assert (jwtUtils.validateToken(accessToken));
        assert (jwtUtils.getUserEmail(accessToken).equals("email"));
    }

    @Test
    public void testGenerateRefreshToken(){
        //given
        //when
        String accessToken = jwtUtils.generateRefreshToken(userAuthInfoDto);
        //then
        assert (jwtUtils.validateToken(accessToken));
        assert (jwtUtils.getUserEmail(accessToken).equals("email"));
    }

    @Test
    public void testValidateToken(){
        //given
        String fakeToken = "fakeToken";
        String realToken = jwtUtils.generateAccessToken(userAuthInfoDto);
        //when
        boolean failureExpected = jwtUtils.validateToken(fakeToken);
        boolean successExpected = jwtUtils.validateToken(realToken);
        // then
        assert (!failureExpected);
        assert (successExpected);
    }
}