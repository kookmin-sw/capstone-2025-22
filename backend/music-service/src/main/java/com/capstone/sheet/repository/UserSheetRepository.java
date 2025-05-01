package com.capstone.sheet.repository;

import com.capstone.sheet.entity.UserSheet;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserSheetRepository extends JpaRepository<UserSheet, Integer> {
    @Query("select u from UserSheet u where u.userEmail=:email")
    List<UserSheet> findAllByEmail(@Param("email") String email);

    List<UserSheet> findAllBySheetName(String sheetName);
}
