package com.capstone.sheet.service;

import com.capstone.exception.InternalServerException;
import com.capstone.sheet.dto.SheetCreateMeta;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class SheetToXmlConverter {
    private void writeFile(Path target, MultipartFile sheetFile) throws IOException {
        try(OutputStream fos = Files.newOutputStream(target)) {
            fos.write(sheetFile.getBytes());
        }
    }

    private void deleteFile(Path targetPath) {
        try {
            if(targetPath != null){
                Files.deleteIfExists(targetPath);
            }
        }catch (IOException e){
            String errorMessage = "failed to delete file: " + e.getMessage();
            log.error(errorMessage);
        }
    }

    private String commandBuilder(String inputPath, String outputPath){
        return String.format("docker run --rm -v %s:/input %s:/output louie8821/audiveris:test", inputPath, outputPath);
    }

    private byte[] processConvert(String inputPath, String outputPath, String fileName){
        ProcessBuilder builder = new ProcessBuilder();
        String command = commandBuilder(inputPath, outputPath);
        builder.command("sh", "-c", command);
        String outputFileName = fileName.replaceFirst("[.][^.]+$", "") + ".xml";
        Path outputFile = null;
        // process docker container
        try{
            Process process = builder.start();
            process.waitFor(30, TimeUnit.SECONDS);
            outputFile = Paths.get(outputPath, outputFileName);
            return Files.readAllBytes(outputFile);
        }catch (IOException | InterruptedException e){
            String errorMessage = "SheetToXmlConverter.convert : " + e.getMessage();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        }finally {
            deleteFile(outputFile);
        }
    }

    public byte[] convertToXml(SheetCreateMeta sheetCreateMeta, MultipartFile sheetFile) {
        Path target = null;
        Path sheetDir = Paths.get("/tmp/sheet/"+ sheetCreateMeta.getUserEmail());
        try {
            Files.createDirectories(sheetDir);
            target = Files.createTempFile(
                    sheetDir,
                    "sheet_",
                    "."+ sheetCreateMeta.getFileExtension());
            writeFile(target, sheetFile);
            return processConvert(target.getParent().toString(), target.getParent() + "/output", target.getFileName().toString());
        }catch (IOException e){
            String errorMessage = "SheetToXmlConverter.convertToXml : " + e.getMessage();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        }finally {
            deleteFile(target);
        }
    }
}
