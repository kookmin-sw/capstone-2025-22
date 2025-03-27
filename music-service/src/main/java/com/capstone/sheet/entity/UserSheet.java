package com.capstone.sheet.entity;

import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@EntityListeners(AuditingEntityListener.class)
public class UserSheet {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int userSheetId;
    @Column
    private String sheetName;
    @Column
    private String color;
    @Column
    private boolean isOwner;
    @CreatedDate
    private LocalDateTime createdDate;
    @Column(nullable = false)
    private String userEmail;
    @ManyToOne
    @JoinColumn(name = "sheet_id")
    private Sheet sheet;
}
