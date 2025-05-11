package com.capstone.practice.repository;

import com.capstone.practice.entity.PatternPractice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PatternPracticeRepository extends JpaRepository<PatternPractice, Integer> {
    List<PatternPractice> findByUserEmail(String userEmail);
}
