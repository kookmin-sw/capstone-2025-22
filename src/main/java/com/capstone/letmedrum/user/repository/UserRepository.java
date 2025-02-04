package com.capstone.letmedrum.user.repository;

import com.capstone.letmedrum.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    @Query("SELECT u FROM User u Where u.email=:email")
    Optional<User> findByEmail(@Param("email") String email);
    @Query("SELECT u FROM User u Where u.nickname=:nickname")
    Optional<User> findByNickname(@Param("nickname") String nickname);
}
