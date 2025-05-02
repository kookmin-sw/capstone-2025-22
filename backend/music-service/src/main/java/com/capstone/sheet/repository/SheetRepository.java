package com.capstone.sheet.repository;

import com.capstone.sheet.entity.Sheet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface SheetRepository extends JpaRepository<Sheet, Integer> {
    @Modifying
    @Query("UPDATE Sheet s SET s.sheetInfo = :newSheetInfo WHERE s.sheetId = :sheetId")
    int updateSheetInfo(@Param("sheetId") int sheetId, @Param("newSheetInfo") byte[] newSheetInfo);
}
