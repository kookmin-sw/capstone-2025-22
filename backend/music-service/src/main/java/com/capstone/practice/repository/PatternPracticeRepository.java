package com.capstone.practice.repository;

import com.capstone.practice.entity.PatternPractice;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PatternPracticeRepository extends JpaRepository<PatternPractice, Integer> {
}
