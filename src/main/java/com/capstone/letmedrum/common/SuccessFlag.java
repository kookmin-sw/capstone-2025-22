package com.capstone.letmedrum.common;

public enum SuccessFlag {
    SUCCESS("success"),
    FAILURE("invalid");
    private final String label;
    SuccessFlag(String label) {
        this.label = label;
    }
}
