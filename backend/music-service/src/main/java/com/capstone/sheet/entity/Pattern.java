package com.capstone.sheet.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Pattern {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column
    private String patternName;

    @Lob
    @Column(columnDefinition = "MEDIUMBLOB")
    private byte[] patternInfo;

    @Column(columnDefinition = "TEXT")
    private String patternJson;
}
