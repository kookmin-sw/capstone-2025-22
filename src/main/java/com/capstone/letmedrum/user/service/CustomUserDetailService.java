    package com.capstone.letmedrum.user.service;

    import com.capstone.letmedrum.config.security.CustomUserDetails;
    import com.capstone.letmedrum.user.entity.User;
    import com.capstone.letmedrum.user.repository.UserRepository;
    import lombok.extern.slf4j.Slf4j;
    import org.springframework.security.core.userdetails.UserDetails;
    import org.springframework.security.core.userdetails.UserDetailsService;
    import org.springframework.stereotype.Service;

    @Service
    @Slf4j
    public class CustomUserDetailService implements UserDetailsService {
        private final UserRepository userRepository;
        public CustomUserDetailService(UserRepository userRepository){
            this.userRepository = userRepository;
        }

        /**
         * load UserDetails by email
         * @param username email, String
         * @return UserDetails - UserDetails, nullable
         * */
        @Override
        public UserDetails loadUserByUsername(String username) {
            User user = userRepository.findByEmail(username).orElse(null);
            return (user==null) ? null : CustomUserDetails
                    .builder()
                    .email(user.getEmail())
                    .password(user.getPassword())
                    .role(user.getRole())
                    .build();
        }
    }
