package com.capstone.practice.repository;

import com.capstone.practice.entity.SheetPractice;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface SheetPracticeRepository extends JpaRepository<SheetPractice, Integer> {
    @Query("select sp " +
            "from SheetPractice sp " +
            "where sp.userEmail=:email and sp.userSheet.userSheetId=:sheetId order by sp.createdDate DESC")
    List<SheetPractice> findAllByEmailAndSheetId(@Param("email") String email, @Param("sheetId") Integer userSheetId, Pageable pageable);
    @Query("select sp " +
            "from SheetPractice sp " +
            "where sp.userSheet.userSheetId=:sheetId order by sp.createdDate DESC limit 1")
    Optional<SheetPractice> findLastPracticeByEmailAndSheetId( @Param("sheetId") Integer userSheetId);
    @Modifying
    @Query("DELETE FROM SheetPractice sp WHERE sp.userSheet.userSheetId = :userSheetId")
    void deletePracticeByUserSheetId(@Param("userSheetId") Integer userSheetId);
    @Query("SELECT MAX(sp.score) FROM SheetPractice sp WHERE sp.userSheet.userSheetId = :userSheetId")
    Optional<Integer> findMaxScoreById(@Param("userSheetId") Integer userSheetId);
}
