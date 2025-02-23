package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.auth.service.AuthManagerService;
import com.capstone.letmedrum.config.security.JwtUtils;
import com.capstone.letmedrum.auth.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.request.UserPasswordUpdateDto;
import com.capstone.letmedrum.user.dto.request.UserProfileUpdateRequestDto;
import com.capstone.letmedrum.user.dto.response.UserProfileUpdateResponseDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.time.ZoneId;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserUpdateServiceTest {
    @InjectMocks
    private UserUpdateService userUpdateService;
    @Mock
    private UserRepository userRepository;
    @Spy
    private JwtUtils jwtUtils = new JwtUtils("eyJhbGciOiJIUzI1NiJ9eyJhbGciOiJIUzI1NiJ9eyJhbGciOiJIUzI1NiJ9eyJhbGciOiJIUzI1NiJ9");
    @Spy
    private PasswordEncoder passwordEncoder;
    @Mock
    private AuthManagerService authManagerService;
    @Test
    void updatePasswordSuccess() {
        // given
        String email = "email@email.com";
        String password = "password";
        String emailToken = jwtUtils.generateJwtToken(
                new UserAuthInfoDto(email, UserRole.ROLE_USER),
                ZoneId.systemDefault(),
                10000L
        );
        UserPasswordUpdateDto updateDto = new UserPasswordUpdateDto(password, emailToken);
        // stub
        when(jwtUtils.validateToken(emailToken)).thenReturn(true);
        when(jwtUtils.getUserEmail(emailToken)).thenReturn(email);
        when(authManagerService.getEmailToken(email)).thenReturn(emailToken);
        when(userRepository.findByEmail(email)).thenReturn(Optional.of(
                new User("email", "password", "nickname", UserRole.ROLE_USER)
        ));
        // when
        userUpdateService.updatePassword(updateDto);
        // then
        verify(jwtUtils, atLeastOnce()).getUserEmail(emailToken);
        verify(passwordEncoder, times(1)).encode(password);
        verify(userRepository, times(1)).findByEmail(email);
    }
    @Test
    void updatePasswordFailed() {
        // given
        String email = "email@email.com";
        String password = "password";
        String emailToken = jwtUtils.generateJwtToken(
                new UserAuthInfoDto(email, UserRole.ROLE_USER),
                ZoneId.systemDefault(),
                10000L
        );
        UserPasswordUpdateDto updateDto = new UserPasswordUpdateDto(password, emailToken);
        // stub
        when(jwtUtils.validateToken(emailToken)).thenReturn(false);
        // then
        assertThrows( RuntimeException.class ,() ->
                // when
                userUpdateService.updatePassword(updateDto)
        );
    }
    @Test
    void updateProfile() {
        String email = "email@email.com";
        String profileImageUrl = "profileImageUrl";
        String nickname = "nickname";
        UserProfileUpdateRequestDto updateDto = new UserProfileUpdateRequestDto(profileImageUrl, nickname);
        String accessToken = jwtUtils.generateAccessToken(
                new UserAuthInfoDto(email, UserRole.ROLE_USER)
        );
        // stub
        when(userRepository.findByEmail(email)).thenReturn(Optional.of(
                new User(email, "password", nickname, UserRole.ROLE_USER)
        ));
        // when
        UserProfileUpdateResponseDto res = userUpdateService.updateProfile(accessToken, updateDto);
        // then
        assertEquals(nickname, res.getNickname());
        assertEquals(profileImageUrl, res.getProfileImage());
    }
}