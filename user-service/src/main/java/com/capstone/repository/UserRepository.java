package com.capstone.repository;

import com.capstone.entity.User;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, String> {
    @Query("SELECT u FROM User u Where u.email=:email")
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    Optional<User> findByEmail(@Param("email") String email);
    @Query("SELECT u FROM User u Where u.nickname=:nickname")
    Optional<User> findByNickname(@Param("nickname") String nickname);
    @Query("UPDATE User u SET u.password=:password WHERE u.email=:email ")
    void updatePassword(@Param("email") String email, @Param("password") String password);
}
