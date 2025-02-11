package com.capstone.letmedrum.common.dto;

import lombok.*;

@Builder
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class CustomResponseDto<T> {
    private String message;
    private T body;
}
