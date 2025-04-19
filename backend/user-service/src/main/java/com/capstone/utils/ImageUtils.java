package com.capstone.utils;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class ImageUtils {
    public static byte[] resizeImage(byte[] image, int maxLength) {
        try {
            // 1. byte[]를 BufferedImage로 변환
            BufferedImage originalImage = ImageIO.read(new ByteArrayInputStream(image));

            // 2. 비율에 맞춰 높이 조정
            double ratio = (double) originalImage.getHeight() / originalImage.getWidth();
            int newWidth, newHeight;
            if(originalImage.getWidth() > originalImage.getHeight()) {
                newWidth = maxLength;
                newHeight = (int) (maxLength * ratio);
            }else{
                newHeight = maxLength;
                newWidth = (int) (maxLength / ratio);
            }

            // 3. 이미지 크기 조정
            BufferedImage resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resizedImage.createGraphics();
            g.drawImage(originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH), 0, 0, null);
            g.dispose();

            // 4. ByteArrayOutputStream을 사용하여 byte[]로 변환
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageOutputStream ios = ImageIO.createImageOutputStream(baos);
            ImageWriter writer = ImageIO.getImageWritersByFormatName("jpg").next();
            ImageWriteParam param = writer.getDefaultWriteParam();

            // 압축 품질 설정
            param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
            param.setCompressionQuality(0.7f); // 예: 70% 품질

            writer.setOutput(ios);
            writer.write(null, new IIOImage(resizedImage, null, null), param);
            ios.close();
            writer.dispose();

            // 최종적으로 byte[] 반환
            return baos.toByteArray();
        } catch (IOException e) {
            throw new InternalError(e.getMessage());
        }
    }
}
