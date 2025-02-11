package com.capstone.letmedrum.common.service;

public interface RedisSingleDataService {
    boolean setValue(String key, String value, int expire);
    String getValue(String key);
    boolean deleteValue(String key);
}
