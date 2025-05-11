package com.capstone.sheet.utils;

import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Slf4j
public class FieldLister {
    public static List<String> getFieldValues(Class<?> clazz) {
        return Stream.of(clazz.getFields())
                .map(field -> {
                    try {
                        return field.get(null) != null ? field.get(null).toString() : null;
                    } catch (IllegalAccessException e) {
                        log.error("Unable to access field: {}", field.getName(), e);
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
}
