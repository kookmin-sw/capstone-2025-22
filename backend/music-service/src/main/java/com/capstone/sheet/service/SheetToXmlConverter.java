package com.capstone.sheet.service;

import com.capstone.exception.InternalServerException;
import com.capstone.sheet.dto.PatternCreateDto;
import com.capstone.sheet.dto.SheetCreateMeta;
import lombok.extern.slf4j.Slf4j;
import org.apache.tomcat.util.http.fileupload.FileUtils;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class SheetToXmlConverter {
    /**
     * write sheetFile's content on target file
     * @param target file to write sheetFile's content
     * @param fileBytes multipart file about sheet content
     * @throws IOException if failed to write file
    * */
    private void writeFile(Path target, byte[] fileBytes) throws IOException {
        try(OutputStream fos = Files.newOutputStream(target)) {
            fos.write(fileBytes);
            fos.flush();
        }
    }

    /**
     * A function that deletes directories even if there are files inside the directory
     * @param targetPath directory to delete
    * */
    private void deleteDirectory(Path targetPath) {
        try {
            if(targetPath != null){
                FileUtils.deleteDirectory(targetPath.toFile());
            }
        }catch (IOException e){
            String errorMessage = "failed to delete file: " + e.getMessage();
            log.error(errorMessage);
        }
    }

    /**
     * Functions that generate docker commands based on input/output paths
     * @param inputPath path of input file
     * @param outputPath path to save output files (mxl, xml)
    * */
    private String[] commandBuilder(String inputPath, String outputPath){
        return new String[]{
                "docker", "run", "--rm", "--privileged",
                "-v", String.format("%s:/input:rw", inputPath),
                "-v", String.format("%s:/output:rw", outputPath),
//                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "--user", "root",
                "louie8821/audiveris:drum"};
    }

    private void consumeStream(InputStream stream) {
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.info("[Docker] {}", line);
                }
            } catch (IOException e) {
                log.warn("Failed to consume stream: {}", e.getMessage());
            }
        }).start();
    }

    /**
     * Function to execute docker instructions and return results
     * @param inputPath path of input file
     * @param outputPath path to save output files (mxl, xml)
     * @return byte array of xml file
    * */
    public byte[] processConvert(String inputPath, String outputPath){
        ProcessBuilder builder = new ProcessBuilder();
        builder.redirectErrorStream(true);
        builder.command(commandBuilder(inputPath, outputPath));
        log.info(Arrays.toString(builder.command().toArray()));
        String outputFileName = "output.xml";
        Path outputFile;
        try{
            Process process = builder.start();
            consumeStream(process.getInputStream());  // stdout
            process.waitFor(5, TimeUnit.MINUTES);
//            process.waitFor();
            outputFile = Paths.get(outputPath, outputFileName);
            return Files.readAllBytes(outputFile);
        }catch (IOException | InterruptedException e){
            String errorMessage = "SheetToXmlConverter.convert : " + e.getMessage();
            e.printStackTrace();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        }
    }

    public String getBasePath(){
        String os = System.getProperty("os.name").toLowerCase();
        String basePath;

        if (os.contains("win")) {
            basePath = "C:\\data"; // 또는 도커 볼륨 마운트 경로에 맞춰서 지정
        } else {
            basePath = "/data";
        }
        return basePath;
    }

    public byte[] cleanMusicXml(byte[] rawBytes) throws UnsupportedEncodingException {
        // 1. byte[] → String (인코딩은 일반적으로 UTF-8, 필요 시 조정)
        String xmlString = new String(rawBytes, StandardCharsets.UTF_8);

        // 2. </score-partwise> 위치까지 잘라냄
        int endIdx = xmlString.indexOf("</score-partwise>");
        if (endIdx != -1) {
            xmlString = xmlString.substring(0, endIdx + "</score-partwise>".length());
        }

        // 3. String → byte[] 재변환
        return xmlString.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Score file life cycle management and xml conversion functions
     * @param sheetCreateMeta metadata of sheet
     * @param fileBytes sheet file data
     * @return byte array of sheet xml
    * */
    public byte[] convertToXml(SheetCreateMeta sheetCreateMeta, byte[] fileBytes) throws InternalServerException {
        Path target;
        String tmpDir = getBasePath();
        Path sheetDir = Paths.get(tmpDir, "sheet", sheetCreateMeta.getUserEmail());
        Path inputDirPath = Paths.get(sheetDir.toString(), "input");
        Path outputDirPath = Paths.get(sheetDir.toString(), "output");

        try {
            // 디렉토리 생성 확인
            Files.createDirectories(inputDirPath);
            Files.createDirectories(outputDirPath);

            // 입력 파일 저장
            target = inputDirPath.resolve("input." + sheetCreateMeta.getFileExtension());
            writeFile(target, fileBytes);

            log.info("Input file exists: {} (size: {})", Files.exists(target), Files.size(target));
            String inputPath = inputDirPath.toAbsolutePath().toString();
            String outputPath = outputDirPath.toAbsolutePath().toString();
            byte[] rawBytes = processConvert(inputPath, outputPath);
            return cleanMusicXml(rawBytes);
        } catch (IOException e) {
            String errorMessage = "SheetToXmlConverter.convertToXml : " + e.getMessage();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        } finally {
            deleteDirectory(sheetDir);
        }
    }

    public byte[] convertToXml(PatternCreateDto createDto, byte[] fileBytes){
        SheetCreateMeta meta = SheetCreateMeta.builder()
                .sheetName(createDto.getPatternName())
                .fileExtension(createDto.getFileExtension())
                .userEmail(UUID.randomUUID().toString().substring(0, 8))
                .build();
        return convertToXml(meta, fileBytes);
    }

}
