package com.capstone.letmedrum.mail.service;

import com.capstone.letmedrum.mail.dto.MailDto;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class CustomMailSenderService {
    private final JavaMailSender javaMailSender;
    /**
     * constructor for DI
     * @param javaMailSender JavaMailSenderImpl
    * */
    public CustomMailSenderService(JavaMailSender javaMailSender){
        this.javaMailSender = javaMailSender;
    }
    /**
     * send mail just with text (plain txt or html)
     * @param mailDto MailDto Object (from, to, txt)
     * @return true if success - not null
     * @throws RuntimeException if MessagingException occur
    * */
    public boolean sendText(MailDto mailDto, boolean isHtml) {
        MimeMessage mimeMessage = javaMailSender.createMimeMessage();
        MimeMessageHelper mimeMessageHelper = new MimeMessageHelper(mimeMessage);
        try {
            mimeMessageHelper.setTo(mailDto.getTo());
            mimeMessageHelper.setFrom(mailDto.getFrom());
            mimeMessageHelper.setText(mailDto.getText(), isHtml);
            javaMailSender.send(mimeMessage);
            return true;
        }catch (MessagingException e){
            log.error(e.getMessage());
            throw new RuntimeException(e);
        }
    }
}
