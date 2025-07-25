package com.capstone.practice.entity;

import com.capstone.sheet.entity.Pattern;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@EntityListeners(AuditingEntityListener.class)
public class PatternPractice {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long patternPracticeId;

    @Column
    private String score;

    @Column(columnDefinition = "Text")
    private String practiceInfo;

    @CreatedDate
    private LocalDateTime createdDate;

    @Column(nullable = false)
    private String userEmail;

    @ManyToOne
    @JoinColumn(name = "pattern_id", referencedColumnName = "id")
    private Pattern pattern;
}
