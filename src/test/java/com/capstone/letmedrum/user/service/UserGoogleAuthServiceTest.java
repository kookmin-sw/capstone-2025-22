package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.dto.GoogleTokenResponseDto;
import com.capstone.letmedrum.auth.dto.GoogleUserInfoResponseDto;
import com.capstone.letmedrum.auth.service.UserAuthService;
import com.capstone.letmedrum.auth.service.UserGoogleAuthService;
import com.capstone.letmedrum.auth.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.repository.UserRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserGoogleAuthServiceTest {
    private MockWebServer mockWebServer;
    @Spy
    @InjectMocks
    private UserGoogleAuthService userGoogleAuthService;
    @Mock
    private UserRepository userRepository;
    @Mock
    private UserAuthService userAuthService;
    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();
    }
    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }
    @Test
    @DisplayName("구글 인증 토큰 발급 단위 테스트")
    void testGetGoogleAccessToken() throws JsonProcessingException, InterruptedException {
        // given
        String accessToken = "access_token";
        String mockUrl = "/google/auth/token";
        GoogleTokenResponseDto response = new GoogleTokenResponseDto(0, accessToken, "", "", "");
        ReflectionTestUtils.setField(userGoogleAuthService, "googleAccessTokenUri", mockWebServer.url(mockUrl).toString());
        mockWebServer.enqueue(new MockResponse()
                .setBody(new ObjectMapper().writeValueAsString(response))
                .setResponseCode(200)
                .addHeader("Content-Type", "application/json"));
        // when
        String result = userGoogleAuthService.getGoogleAccessToken(accessToken);
        RecordedRequest recordedRequest = mockWebServer.takeRequest();
        // then
        System.out.println(accessToken);
        System.out.println(recordedRequest.getMethod());
        System.out.println(recordedRequest.getPath());
        assertEquals(accessToken, result);
        assertEquals("POST", recordedRequest.getMethod());
        assertEquals(mockUrl, recordedRequest.getPath());
    }
    @Test
    @DisplayName("구글 사용자 정보 발급 단위 테스트")
    void testGetGoogleUserInfo() throws JsonProcessingException, InterruptedException {
        //given
        String email = "userEmail", accessToken = "access_token";
        String mockUrl = "/google/auth/userinfo";
        GoogleUserInfoResponseDto response = GoogleUserInfoResponseDto.builder().email(email).build();
        ReflectionTestUtils.setField(userGoogleAuthService, "googleUserInfoUri", mockWebServer.url(mockUrl).toString());
        mockWebServer.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", "application/json")
                .setBody(new ObjectMapper().writeValueAsString(response)));
        //when
        GoogleUserInfoResponseDto result = userGoogleAuthService.getGoogleUserInfo(accessToken);
        RecordedRequest recordedRequest = mockWebServer.takeRequest();
        //then
        System.out.println(email);
        System.out.println(recordedRequest.getMethod());
        System.out.println(recordedRequest.getPath());
        assertEquals(email, result.getEmail());
        assertEquals("GET", recordedRequest.getMethod());
        assertEquals(mockUrl, recordedRequest.getPath());
    }
    @Test
    @DisplayName("구글 로그인 단위 테스트")
    void testGoogleLogin() {
        // given
        String email = "userEmail";
        UserAuthResponseDto expectedResult = UserAuthResponseDto.builder().email(email).build();
        GoogleUserInfoResponseDto expectedUserInfo = GoogleUserInfoResponseDto.builder().email(email).build();
        User savedUser = User.builder().email(email).build();
        // stub
        doReturn("access_token").when(userGoogleAuthService).getGoogleAccessToken(anyString());
        doReturn(expectedUserInfo).when(userGoogleAuthService).getGoogleUserInfo(anyString());
        when(userRepository.findByEmail(anyString())).thenReturn(Optional.of(savedUser));
        when(userAuthService.generateResponseAndSaveToken(anyString(), any())).thenReturn(expectedResult);
        // when
        UserAuthResponseDto res = userGoogleAuthService.signIn("authCode");
        // then
        System.out.println(res.getEmail());
        assertEquals(expectedResult.getEmail(), res.getEmail());
    }
}