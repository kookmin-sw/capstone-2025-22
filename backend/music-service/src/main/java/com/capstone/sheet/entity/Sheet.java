package com.capstone.sheet.entity;

import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
@EntityListeners(AuditingEntityListener.class)
public class Sheet {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int sheetId;

    @Lob
    @Column(columnDefinition = "MEDIUMBLOB")
    private byte[] sheetInfo;

    @Column
    private String sheetJson;

    @Column
    private String author;

    @CreatedDate
    private LocalDateTime createdDate;

    public static Sheet create(){
        return Sheet.builder().build();
    }
}
