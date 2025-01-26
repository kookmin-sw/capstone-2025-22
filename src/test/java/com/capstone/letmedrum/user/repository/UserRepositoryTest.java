package com.capstone.letmedrum.user.repository;

import com.capstone.letmedrum.user.entity.User;
import com.capstone.letmedrum.user.entity.UserRole;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class UserRepositoryTest {
    @Autowired
    private UserRepository userRepository;
    public void init(){
        userRepository.deleteAll();
    }
    @Test
    public void testFindByEmail(){
        // given
        init();
        String email = "email";
        User user = User.builder()
                .email(email)
                .password("password")
                .role(UserRole.ROLE_GUEST)
                .nickname("nickname")
                .build();
        // when
        userRepository.save(user);
        User existUser = userRepository.findByEmail(email).orElse(null);
        // then
        assert (existUser != null);
        assert (existUser.getEmail().equals(email));
    }

}