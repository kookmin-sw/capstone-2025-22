package com.capstone.practice.repository;

import com.capstone.practice.entity.SheetPractice;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SheetPracticeRepository extends JpaRepository<SheetPractice, Integer> {
    @Query("select sp " +
            "from SheetPractice sp " +
            "where sp.userEmail=:email and sp.userSheet.userSheetId=:sheetId order by sp.createdDate DESC")
    List<SheetPractice> findAllByEmailAndSheetId(@Param("email") String email, @Param("sheetId") Integer userSheetId, Pageable pageable);
    @Query("select sp " +
            "from SheetPractice sp " +
            "where sp.userEmail=:email and sp.userSheet.userSheetId=:sheetId order by sp.createdDate DESC limit 1")
    SheetPractice findLastPracticeByEmailAndSheetId(@Param("email") String email, @Param("sheetId") Integer userSheetId);
}
