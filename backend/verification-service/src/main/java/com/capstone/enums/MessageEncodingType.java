package com.capstone.enums;

public enum MessageEncodingType {
    UTF_8("UTF-8");
    private final String label;
    MessageEncodingType(String label){
        this.label = label;
    }
}
