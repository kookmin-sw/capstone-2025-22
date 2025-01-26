package com.capstone.letmedrum.user.entity;


import lombok.Getter;

@Getter
public enum UserRole {
    ROLE_GUEST("ROLE_GUEST"),
    ROLE_USER("ROLE_USER"),
    ROLE_ADMIN("ROLE_ADMIN");
    private final String label;
    UserRole(String label) {
        this.label = label;
    }
}
