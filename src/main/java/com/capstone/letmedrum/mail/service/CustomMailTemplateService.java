package com.capstone.letmedrum.mail.service;

import com.capstone.letmedrum.mail.dto.AuthCodeTemplateDto;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.thymeleaf.context.Context;
import org.thymeleaf.spring6.SpringTemplateEngine;

@Slf4j
@Service
public class CustomMailTemplateService {
    private final SpringTemplateEngine templateEngine;
    private final String authCodeTemplate = "customMailTemplate";
    /**
     * constructor
     * @param templateEngine SpringTemplateEngine class
    * */
    public CustomMailTemplateService(SpringTemplateEngine templateEngine) {
        this.templateEngine = templateEngine;
    }
    /**
     * generate auth code email html template (String)
     * @param authCodeTemplateDto AuthCodeTemplateDto( title, content, authCode)
     * @return html template - String, not-null
    * */
    public String generateAuthCodeTemplate(AuthCodeTemplateDto authCodeTemplateDto) {
        Context context = new Context();
        context.setVariable("title", authCodeTemplateDto.getTitle());
        context.setVariable("content", authCodeTemplateDto.getContent());
        context.setVariable("authCode", authCodeTemplateDto.getAuthCode());
        return templateEngine.process(authCodeTemplate, context);
    }
}
