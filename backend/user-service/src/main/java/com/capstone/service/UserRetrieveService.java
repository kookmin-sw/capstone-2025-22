package com.capstone.service;

import com.capstone.entity.User;
import com.capstone.exception.InvalidUserInfoException;
import com.capstone.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class UserRetrieveService {
    private final UserRepository userRepository;
    public UserRetrieveService(UserRepository userRepository){
        this.userRepository = userRepository;
    }
    /**
     * if user with nickname exists return user, else throw exception
     * @param nickname - user's nickname, String
     * @return User - User, nullable
    * */
    public User getUserOrExceptionByNickname(String nickname){
        return userRepository.findByNickname(nickname)
                .orElseThrow(()->new InvalidUserInfoException("user not exists"));
    }
    /**
     *  if user with email exists return user, else throw exception
     * @param email - user's email, String
     * @return User - User, not-null
     * @throws InvalidUserInfoException if user not exists
    * */
    @Transactional
    public User getUserOrExceptionByEmail(String email){
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new InvalidUserInfoException("user not exists"));
    }
    /**
     *  if user with email exists return user, else return null
     * @param email - user's email
     * @return User - User, nullable
     */
    public User getUserOrNullByEmail(String email){
        return userRepository.findByEmail(email)
                .orElse(null);
    }
}
