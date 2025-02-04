package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserAuthServiceTest {
    @InjectMocks
    private UserAuthService userAuthService;
    @Mock
    private UserRepository userRepository;
    @Mock
    private PasswordEncoder passwordEncoder;
    @Mock
    private JwtUtils jwtUtils;
    @Test
    @DisplayName("회원가입 성공 테스트")
    void testSignUpUserSuccess() {
        // given
        String email = "email";
        UserCreateDto userCreateDto = UserCreateDto
                .builder()
                .email(email)
                .nickname("nickname")
                .password("password")
                .role(UserRole.ROLE_GUEST)
                .build();
        // stub
        when(userRepository.findByEmail(any(String.class)))
                .thenReturn(Optional.empty());
        when(userRepository.save(any(User.class)))
                .thenReturn(userCreateDto.toEntity());
        // when
        UserAuthResponseDto res = userAuthService.signUpUser(userCreateDto);
        // then
        assert (res.getEmail().equals(email));
    }
    @Test
    @DisplayName("회원가입 실패 테스트")
    void testSignUpUserFailure(){
        // given
        String email = "email", password = "password";
        UserCreateDto userCreateDto = UserCreateDto.builder()
                .email(email)
                .password(password)
                .role(UserRole.ROLE_USER)
                .build();
        // stub
        when(userRepository.findByEmail(anyString()))
                .thenReturn(Optional.of(userCreateDto.toEntityWithEncodedPassword(new BCryptPasswordEncoder())));
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.signUpUser(userCreateDto));
    }
    @Test
    @DisplayName("로그인 성공 테스트")
    void testSignInUserSuccess() {
        // given
        String email = "email";
        String password = "password";
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(email)
                .password(password)
                .role(UserRole.ROLE_USER)
                .build();
        UserCreateDto userCreateDto = UserCreateDto
                .builder()
                .email(email)
                .nickname("")
                .password(password)
                .role(UserRole.ROLE_USER)
                .build();
        // stub
        when(passwordEncoder.matches(anyString(), anyString()))
                .thenReturn(true);
        when(userRepository.findByEmail(anyString()))
                .thenReturn(Optional.of(userCreateDto.toEntityWithEncodedPassword(new BCryptPasswordEncoder())));
        // when
        UserAuthResponseDto signInRes = userAuthService.signInUser(userAuthInfoDto);
        // then
        assert (signInRes.getEmail().equals(email));
    }
    @Test
    @DisplayName("로그인 실패 테스트")
    void testSignInUserFailure(){
        // given
        String email = "email", password = "password";
        String wrongEmail = "wrongEmail", wrongPassword = "wrongPassword";
        UserCreateDto userCreateDto = UserCreateDto.builder()
                .email(email)
                .password(password)
                .nickname("")
                .role(UserRole.ROLE_USER).build();
        UserAuthInfoDto wrongEmailPasswordUser = UserAuthInfoDto.builder()
                .email(wrongEmail)
                .password(wrongPassword)
                .role(UserRole.ROLE_USER).build();
        // stub
        when(userRepository.findByEmail(anyString()))
                .thenReturn(Optional.of(userCreateDto.toEntityWithEncodedPassword(new BCryptPasswordEncoder())));
        when(passwordEncoder.matches(anyString(), anyString()))
                .thenReturn(false);
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.signInUser(wrongEmailPasswordUser));
    }
}