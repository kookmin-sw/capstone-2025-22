package com.capstone.sheet.entity;

import jakarta.persistence.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@EntityListeners(AuditingEntityListener.class)
public class Sheet {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int sheetId;

    @Column(columnDefinition = "Text")
    private String sheetInfo;

    @CreatedDate
    private LocalDateTime createdDate;
}
