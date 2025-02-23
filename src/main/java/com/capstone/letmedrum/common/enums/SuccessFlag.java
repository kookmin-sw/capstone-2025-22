package com.capstone.letmedrum.common.enums;

public enum SuccessFlag {
    SUCCESS("valid"),
    FAILURE("invalid");
    private final String label;
    SuccessFlag(String label) {
        this.label = label;
    }
    public String getLabel() {
        return label;
    }
}
