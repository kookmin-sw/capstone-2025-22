package com.capstone.sheet.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;

@Entity
public class Pattern {
    @Id
    private String id;
    @Column
    private String patternName;
    @Column(columnDefinition = "Text")
    private String patternInfo;
}
