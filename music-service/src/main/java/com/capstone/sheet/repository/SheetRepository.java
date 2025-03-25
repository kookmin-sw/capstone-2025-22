package com.capstone.sheet.repository;

import com.capstone.sheet.entity.Sheet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SheetRepository extends JpaRepository<Sheet, Integer> {
}
