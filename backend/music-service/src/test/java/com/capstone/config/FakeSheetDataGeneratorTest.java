package com.capstone.config;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@ActiveProfiles({"test", "dev"})
class FakeSheetDataGeneratorTest {
    @Autowired
    private FakeSheetDataGenerator fakeSheetDataGenerator;

    @Test
    void getPracticeInfo_success(){
        assert true;
    }
}