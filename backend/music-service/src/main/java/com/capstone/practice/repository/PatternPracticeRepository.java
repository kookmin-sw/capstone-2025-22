package com.capstone.practice.repository;

import com.capstone.practice.entity.PatternPractice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatternPracticeRepository extends JpaRepository<PatternPractice, Integer> {

    @Query("SELECT pp FROM PatternPractice pp WHERE pp.pattern.id = :patternId AND pp.userEmail=:userEmail AND  pp.score=(" +
            "select MAX(tpp.score) from PatternPractice tpp where tpp.pattern.id=pp.pattern.id )")
    List<PatternPractice> findMaxScorePracticesByPatternId(Long patternId, String userEmail);

    @Query("SELECT pp FROM PatternPractice pp WHERE pp.pattern.id = :patternId AND pp.userEmail=:userEmail")
    List<PatternPractice> findByPatternAndUserEmail(Long patternId, String userEmail);
}
