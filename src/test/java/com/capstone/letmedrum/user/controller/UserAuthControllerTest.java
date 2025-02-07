package com.capstone.letmedrum.user.controller;

import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.service.UserAuthService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserAuthControllerTest {
    @InjectMocks
    private UserAuthController userAuthController;
    @Mock
    private UserAuthService userAuthService;
    private MockMvc mockMvc;
    @BeforeEach
    void init(){
        mockMvc = MockMvcBuilders.standaloneSetup(userAuthController).build();
    }
    @Test
    @DisplayName("로그인 테스트")
    void testSignInUser() throws Exception {
        // given
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto.builder()
                .email("email")
                .role(UserRole.ROLE_GUEST)
                .build();
        String content = new ObjectMapper().writeValueAsString(userAuthInfoDto);
        // when
        ResultActions resultActions = mockMvc.perform(
                MockMvcRequestBuilders.post("/auth/signin")
                        .content(content)
                        .contentType(MediaType.APPLICATION_JSON)
        );
        // then
        resultActions
                .andExpect(MockMvcResultMatchers.status().isOk())
                .andReturn();
    }
    @Test
    @DisplayName("회원가입 테스트")
    void signUpUser() throws Exception{
        // given
        UserCreateDto userCreateDto = UserCreateDto.builder()
                .email("email")
                .password("password")
                .nickname("nickname")
                .build();
        String content = new ObjectMapper().writeValueAsString(userCreateDto);
        // when
        ResultActions resultActions = mockMvc.perform(
                MockMvcRequestBuilders.post("/auth/signup")
                        .content(content)
                        .contentType(MediaType.APPLICATION_JSON)
        );
        // then
        resultActions
                .andExpect(MockMvcResultMatchers.status().isOk())
                .andReturn();
    }
}