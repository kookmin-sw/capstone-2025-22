package com.capstone.letmedrum.user.service;

import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.dto.UserAuthResponseDto;
import com.capstone.letmedrum.user.dto.UserCreateDto;
import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import com.capstone.letmedrum.user.repository.UserRepository;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class UserAuthServiceTest {
    @Autowired
    private UserAuthService userAuthService;
    @Autowired
    private UserRepository userRepository;
    public void init(){
        userRepository.deleteAll();
    }
    @Test
    void TestSignInUser() {
        // given
        init();
        String email = "email";
        String password = "password";
        UserAuthInfoDto userAuthInfoDto = UserAuthInfoDto
                .builder()
                .email(email)
                .password(password)
                .role(UserRole.ROLE_GUEST)
                .build();
        UserCreateDto userCreateDto = UserCreateDto
                .builder()
                .email(email)
                .nickname("")
                .password(password)
                .role(UserRole.ROLE_GUEST)
                .build();
        // when
        UserAuthResponseDto signUpRes = userAuthService.signUpUser(userCreateDto);
        UserAuthResponseDto signInRes = userAuthService.signInUser(userAuthInfoDto);
        // then
        assert (signInRes!=null && signUpRes!=null);
        assert (signInRes.getEmail().equals(email));
    }
    @Test
    void TestSignUpUser() {
        // given
        init();
        String email = "email";
        UserCreateDto userCreateDto1 = UserCreateDto
                .builder()
                .email(email)
                .nickname("nickname")
                .password("password")
                .role(UserRole.ROLE_GUEST)
                .build();
        UserCreateDto userCreateDto2 = UserCreateDto
                .builder()
                .email(email)
                .nickname("nickname")
                .password("password")
                .role(UserRole.ROLE_GUEST)
                .build();
        // when
        UserAuthResponseDto res1 = userAuthService.signUpUser(userCreateDto1);
        UserAuthResponseDto res2 = userAuthService.signUpUser(userCreateDto2);
        // then
        assert (res1.getEmail().equals(email));
        assert (res2==null);
    }
}