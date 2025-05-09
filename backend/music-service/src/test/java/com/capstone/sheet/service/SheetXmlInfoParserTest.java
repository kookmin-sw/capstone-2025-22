package com.capstone.sheet.service;

import com.capstone.dto.musicXml.PartInfo;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.test.context.ActiveProfiles;

import java.nio.file.Files;
import java.util.List;

@SpringBootTest
@ActiveProfiles("test")
class SheetXmlInfoParserTest {
    @Autowired
    private SheetXmlInfoParser sheetXmlInfoParser;

    @Test
    void parseXmlInfo() throws Exception{
        // given
        ResourceLoader resourceLoader = new DefaultResourceLoader();
        Resource sheetXmlInfo = resourceLoader.getResource("classpath:sheet/sheet.xml");
        byte[] sheetXmlBytes = Files.readAllBytes(sheetXmlInfo.getFile().toPath());
        // when
        List<PartInfo> partInfoList = sheetXmlInfoParser.parseXmlInfo(sheetXmlBytes);
        // then
        assert partInfoList.size() == 1;
        assert partInfoList.get(0).getMeasureList().size() == 76;
    }
}