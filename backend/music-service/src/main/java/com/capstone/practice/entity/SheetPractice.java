package com.capstone.practice.entity;

import com.capstone.dto.sheet.MusicServiceClientDto;
import com.capstone.sheet.entity.Sheet;
import com.capstone.sheet.entity.UserSheet;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
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
    @JoinColumn(name = "user_sheet_id")
    private UserSheet userSheet;

    public static SheetPractice from(
            MusicServiceClientDto.SheetPracticeCreateRequest sheetPracticeCreateRequest,
            UserSheet userSheet) throws JsonProcessingException {
        String practiceInfoString = new ObjectMapper().writeValueAsString(sheetPracticeCreateRequest.getFinalMeasures());
        return SheetPractice.builder()
                .score((int) sheetPracticeCreateRequest.getScore())
                .userEmail(sheetPracticeCreateRequest.getUserEmail())
                .userSheet(userSheet)
                .practiceInfo(practiceInfoString)
                .build();
    }
}
