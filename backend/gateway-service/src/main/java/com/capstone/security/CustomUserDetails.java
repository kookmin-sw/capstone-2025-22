package com.capstone.security;

import com.capstone.auth.UserAuthInfoDto;
import com.capstone.enums.UserRole;
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
    @Builder
    public CustomUserDetails(String email, String password, UserRole role){
        this.userAuthInfoDto = UserAuthInfoDto.builder()
                .email(email)
                .role(role)
                .build();
    }
    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        List<String> roles = new ArrayList<>();
        if(userAuthInfoDto.getRole() != null) {
            roles.add(userAuthInfoDto.getRole().name());
        }
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
