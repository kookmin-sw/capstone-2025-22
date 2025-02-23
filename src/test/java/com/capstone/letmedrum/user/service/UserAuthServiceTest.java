package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.service.AuthManagerService;
import com.capstone.letmedrum.auth.service.UserAuthService;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.auth.dto.UserAuthInfoDto;
import com.capstone.letmedrum.auth.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.request.UserCreateDto;
import com.capstone.letmedrum.auth.dto.UserSignInDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserAuthServiceTest {
    @Spy
    @InjectMocks
    private UserAuthService userAuthService;
    @Mock
    private UserRepository userRepository;
    @Mock
    private PasswordEncoder passwordEncoder;
    @Mock
    private JwtUtils jwtUtils;
    @Mock
    private AuthManagerService authManagerService;
    @Test
    @DisplayName("회원가입 성공 테스트")
    void testSignUpUserSuccess() {
        // given
        String email = "email";
        String nickname = "username";
        String password = "password";
        UserCreateDto userCreateDto = UserCreateDto.builder()
                .email(email)
                .nickname(nickname)
                .password(password).build();
        // stub
        when(userRepository.findByEmail(email))
                .thenReturn(Optional.empty());
        when(userRepository.save(any(User.class)))
                .thenReturn(userCreateDto.toEntity());
        when(userAuthService.generateResponseAndSaveToken(email, UserRole.ROLE_USER))
                .thenReturn(UserAuthResponseDto.builder().email(email).build());
        // when
        UserAuthResponseDto res = userAuthService.signUpUser(userCreateDto);
        // then
        assert (res.getEmail().equals(email));
    }
    @Test
    @DisplayName("회원가입 실패 테스트 : 이미 가입된 사용자")
    void testSignUpUserFailure(){
        // given
        String email = "email";
        String password = "password";
        UserCreateDto userCreateDto = UserCreateDto.builder()
                .email(email)
                .password(password)
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
        UserSignInDto userSignInDto = new UserSignInDto(email, password);
        UserCreateDto userCreateDto = UserCreateDto
                .builder()
                .email(email)
                .password(password)
                .build();
        // stub
        when(passwordEncoder.matches(any(), anyString()))
                .thenReturn(true);
        when(userRepository.findByEmail(email))
                .thenReturn(Optional.of(userCreateDto.toEntityWithEncodedPassword(new BCryptPasswordEncoder())));
        when(userAuthService.generateResponseAndSaveToken(email, UserRole.ROLE_USER))
                .thenReturn(UserAuthResponseDto.builder().email(email).build());
        // when
        UserAuthResponseDto signInRes = userAuthService.signInUser(userSignInDto);
        // then
        assert (signInRes.getEmail().equals(email));
    }
    @Test
    @DisplayName("로그인 실패 테스트 : 정보 불일치")
    void testSignInUserFailure(){
        // given
        String wrongEmail = "wrongEmail", wrongPassword = "wrongPassword";
        UserSignInDto wrongEmailPasswordUser = new UserSignInDto(wrongEmail, wrongPassword);
        // stub
        when(userRepository.findByEmail(anyString()))
                .thenReturn(Optional.empty());
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.signInUser(wrongEmailPasswordUser));
    }
    @Test
    @DisplayName("로그아웃 성공 테스트")
    void testSignOutUserSuccess() {
        // given
        String email = "email";
        String secret_key = "abcdefgjijklmnopqrstuvwxyzabcdefgjijklmnopqrstuvwxyz";
        String refresh_token = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("email", UserRole.ROLE_USER)
        );
        User savedUser = new User();
        savedUser.setEmail(email);
        // stub
        when(jwtUtils.validateToken(refresh_token))
                .thenReturn(true);
        when(jwtUtils.getUserEmail(refresh_token))
                .thenReturn(email);
        when(userRepository.findByEmail(email))
                .thenReturn(Optional.of(savedUser));
        // when
        boolean res = userAuthService.signOutUser(refresh_token);
        // then
        assertTrue(res);
    }
    @Test
    @DisplayName("로그아웃 실패 테스트 : 토큰 만료")
    void testSignOutUserFailureInvalidToken() {
        // given
        String secret_key = "abcdefgjijklmnopqrstuvwxyzabcdefgjijklmnopqrstuvwxyz";
        String refresh_token = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("email", UserRole.ROLE_USER)
        );
        // stub
        when(jwtUtils.validateToken(refresh_token))
                .thenReturn(false);
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.signOutUser(refresh_token));
    }
    @Test
    @DisplayName("로그아웃 실패 테스트 : 사용자 없음")
    void testSignOutUserFailureInvalidUser() {
        // given
        String email = "email";
        String secret_key = "abcdefgjijklmnopqrstuvwxyzabcdefgjijklmnopqrstuvwxyz";
        String refresh_token = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("email", UserRole.ROLE_USER)
        );
        // stub
        when(jwtUtils.validateToken(refresh_token))
                .thenReturn(true);
        when(jwtUtils.getUserEmail(anyString()))
                .thenReturn(email);
        when(userRepository.findByEmail(anyString()))
                .thenReturn(Optional.empty());
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.signOutUser(refresh_token));
    }
    @Test
    @DisplayName("토큰 재발급 성공 테스트")
    void refreshTokenSuccess(){
        // given
        String email = "email";
        String secret_key = "abcdefgjijklmnopqrstuvwxyzabcdefgjijklmnopqrstuvwxyz";
        String refresh_token = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("email", UserRole.ROLE_USER)
        );
        User savedUser = new User();
        savedUser.setEmail(email);
        String newToken = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("new email", UserRole.ROLE_USER)
        );
        // stub
        when(jwtUtils.validateToken(refresh_token))
                .thenReturn(true);
        when(jwtUtils.getUserEmail(refresh_token))
                .thenReturn(email);
        when(authManagerService.getRefreshToken(email))
                .thenReturn(refresh_token);
        when(userAuthService.generateResponseAndSaveToken(email, UserRole.ROLE_USER))
                .thenReturn(UserAuthResponseDto.builder().email(email).refreshToken(newToken).build());
        // when
        UserAuthResponseDto res = userAuthService.doRefreshTokenRotation(refresh_token);
        // then
        assert (res.getEmail().equals(email));
        assert (!res.getRefreshToken().equals(refresh_token));
    }
    @Test
    @DisplayName("토큰 재발급 실패 테스트 : 이전 버전 토큰")
    void refreshTokenFailure(){
        // given
        String email = "email";
        String secret_key = "abcdefgjijklmnopqrstuvwxyzabcdefgjijklmnopqrstuvwxyz";
        String currentRefreshToken = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("email", UserRole.ROLE_USER)
        );
        String prevRefreshToken = new JwtUtils(secret_key).generateRefreshToken(
                new UserAuthInfoDto("new email", UserRole.ROLE_USER)
        );
        User savedUser = new User();
        savedUser.setEmail(email);
        // stub
        when(jwtUtils.validateToken(prevRefreshToken))
                .thenReturn(true);
        when(userRepository.findByEmail(email))
                .thenReturn(Optional.of(savedUser));
        when(authManagerService.getRefreshToken(email))
                .thenReturn(currentRefreshToken);
        // when
        // then
        assertThrows(RuntimeException.class, () -> userAuthService.doRefreshTokenRotation(prevRefreshToken));
    }
}