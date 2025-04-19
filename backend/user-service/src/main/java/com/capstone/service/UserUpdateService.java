package com.capstone.service;

import com.capstone.client.AuthClientService;
import com.capstone.dto.request.UserCreateDto;
import com.capstone.dto.request.UserPasswordUpdateDto;
import com.capstone.dto.response.UserInfoDto;
import com.capstone.dto.response.UserProfileUpdateResponseDto;
import com.capstone.auth.PasswordEncoder;
import com.capstone.entity.User;
import com.capstone.exception.*;
import com.capstone.jwt.JwtUtils;
import com.capstone.repository.UserRepository;
import com.capstone.utils.ImageUtils;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpStatus;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.List;

@Service
public class UserUpdateService {
    private final UserRepository userRepository;
    private final JwtUtils jwtUtils;
    private final AuthClientService authClientService;
    private final PasswordEncoder passwordEncoder;
    public UserUpdateService(UserRepository userRepository, JwtUtils jwtUtils, AuthClientService authClientService, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
        this.authClientService = authClientService;
        this.passwordEncoder = passwordEncoder;
    }
    @Transactional
    public void updatePassword(UserPasswordUpdateDto updateDto) {
        if(!jwtUtils.validateToken(updateDto.getEmailToken())){
            throw new InvalidRequestException("Invalid email token : expired or invalid");
        }
        String email = jwtUtils.getUserEmail(updateDto.getEmailToken());
        if(!authClientService.findEmailTokenSync(email).equals(updateDto.getEmailToken())){
            throw new InvalidTokenException("Invalid email token : expired or invalid");
        }
        String encodedPassword = passwordEncoder.encode(updateDto.getNewPassword());
        User user = userRepository.findByEmail(email)
                .orElseThrow(()->new InvalidUserInfoException("User not found"));
        user.setPassword(encodedPassword);
    }
    @Transactional
    public Mono<UserProfileUpdateResponseDto> updateProfile(String accessToken, String nickname, FilePart profileImage) {
        accessToken = jwtUtils.processToken(accessToken);
        if(!jwtUtils.validateToken(accessToken)) throw new InvalidTokenException("Invalid access token : expired or invalid");
        String email = jwtUtils.getUserEmail(accessToken);
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new InvalidUserInfoException("User not found"));
        user.setNickname(nickname);
        return profileImage!=null ?
            profileImage.content()
                .collectList()
                .flatMap(dataBuffers -> {
                    byte[] imageBytes = ImageUtils.resizeImage(mergeDataBuffers(dataBuffers), 300);
                    user.setProfileImage(imageBytes); // 프로필 이미지 설정
                    return Mono.just(userRepository.save(user));
                })
                .flatMap(savedUser -> Mono.just(UserProfileUpdateResponseDto.builder()
                        .profileImage(getBase64Image(savedUser.getProfileImage()))
                        .nickname(savedUser.getNickname())
                        .build()))
            : Mono.just(UserProfileUpdateResponseDto.builder()
                .profileImage(user.getProfileImage()!=null ? getBase64Image(user.getProfileImage()) : "null")
                .nickname(user.getNickname()).build());
    }
    @Transactional
    public UserProfileUpdateResponseDto updateProfile(String accessToken, String nickname, MultipartFile profileIamge){
        accessToken = jwtUtils.processToken(accessToken);
        if(!jwtUtils.validateToken(accessToken)) throw new InvalidTokenException("Invalid access token : expired or invalid");
        String email = jwtUtils.getUserEmail(accessToken);
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new InvalidUserInfoException("User not found"));
        user.setNickname(nickname);
        if(profileIamge!=null){
            try{
                byte[] imageBytes = ImageUtils.resizeImage(profileIamge.getBytes(), 300);
                user.setProfileImage(imageBytes);
            }catch (IOException e){
                throw new InternalServerException(e.getMessage());
            }
        }
        userRepository.save(user);
        UserProfileUpdateResponseDto res = UserProfileUpdateResponseDto.builder()
                .nickname(user.getNickname()).build();
        if(profileIamge!=null) res.setProfileImage(getBase64Image(user.getProfileImage()));
        return res;
    }
    @Transactional
    public UserInfoDto createUser(UserCreateDto userCreateDto) {
        if(userRepository.findByEmail(userCreateDto.getEmail()).isPresent())
            throw new CustomException(HttpStatus.CONFLICT, "user already exists");
        if(userRepository.findByNickname(userCreateDto.getNickname()).isPresent())
            throw new CustomException(HttpStatus.CONFLICT, "nickname duplicated");
        User user = userRepository.save(userCreateDto.toEntity());
        return UserInfoDto.builder()
                .email(user.getEmail())
                .password(user.getPassword())
                .nickname(user.getNickname())
                .role(user.getRole())
                .build();
    }
    private String getBase64Image(byte[] file) {
        return Base64.getEncoder().encodeToString(file);
    }
    private byte[] mergeDataBuffers(List<DataBuffer> dataBuffers) {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try{
            for(DataBuffer buffer : dataBuffers) {
                byte[] bytes = new byte[buffer.readableByteCount()];
                buffer.read(bytes);
                bos.write(bytes);
            }
        }catch (IOException e){
            throw new InternalServerException(e.getMessage());
        }
        return bos.toByteArray();
    }
}
