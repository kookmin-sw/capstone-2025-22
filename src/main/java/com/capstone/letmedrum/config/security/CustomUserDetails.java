package com.capstone.letmedrum.config.security;

import com.capstone.letmedrum.user.dto.UserAuthInfoDto;
import com.capstone.letmedrum.user.entity.UserRole;
import lombok.Builder;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class CustomUserDetails implements UserDetails {
    private final UserAuthInfoDto userAuthInfoDto;
    public CustomUserDetails(UserAuthInfoDto userAuthInfoDto){
        this.userAuthInfoDto = userAuthInfoDto;
    }
    @Builder
    public CustomUserDetails(String email, String password, UserRole role){
        this.userAuthInfoDto = UserAuthInfoDto.builder()
                .email(email)
                .password(password)
                .role(role)
                .build();
    }
    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        List<String> roles = new ArrayList<>();
        roles.add(userAuthInfoDto.getRole().name());
        return roles.stream()
                .map(SimpleGrantedAuthority::new)
                .collect(Collectors.toList());
    }
    @Override
    public String getPassword() {
        return null;
    }
    @Override
    public String getUsername() {
        return null;
    }
}
