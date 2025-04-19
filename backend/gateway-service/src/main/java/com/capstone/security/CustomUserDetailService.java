    package com.capstone.security;

    import com.capstone.enums.UserRole;
    import com.capstone.jwt.JwtUtils;
    import lombok.extern.slf4j.Slf4j;
    import org.springframework.security.core.userdetails.UserDetails;
    import org.springframework.security.core.userdetails.UserDetailsService;
    import org.springframework.stereotype.Service;

    @Service
    @Slf4j
    public class CustomUserDetailService implements UserDetailsService {

        private final JwtUtils jwtUtils;

        public CustomUserDetailService(JwtUtils jwtUtils) {
            this.jwtUtils = jwtUtils;
        }

        /**
         * load UserDetails by email
         * @param userToken email, String
         * @return UserDetails - UserDetails, nullable
         * */
        @Override
        public UserDetails loadUserByUsername(String userToken) {
            String email = jwtUtils.getUserEmail(userToken);
            UserRole role = jwtUtils.getUserRole(userToken);
            return CustomUserDetails.builder()
                    .email(email)
                    .role(role)
                    .build();
        }
    }
