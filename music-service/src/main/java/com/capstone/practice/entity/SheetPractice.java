package com.capstone.practice.entity;

import com.capstone.sheet.entity.Sheet;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@EntityListeners(AuditingEntityListener.class)
public class SheetPractice {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int sheetPracticeId;
    @Column
    private int score;
    @Column(columnDefinition = "Text")
    private String practiceInfo;
    @CreatedDate
    private LocalDateTime createdDate;
    @Column(nullable = false)
    private String userEmail;
    @ManyToOne
    @JoinColumn(name = "sheet_id")
    private Sheet sheet;
}
