package com.capstone.sheet.repository;

import com.capstone.sheet.entity.Pattern;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatternRepository extends JpaRepository<Pattern, Long> {
    @Query("select distinct p from Pattern p join PatternPractice pr on p.id=pr.pattern.id where pr.userEmail=:userEmail and CAST(pr.score AS int)>=:score")
    List<Pattern> findByUserEmailAndScoreGreaterThanEqual(String userEmail, int score);
}
