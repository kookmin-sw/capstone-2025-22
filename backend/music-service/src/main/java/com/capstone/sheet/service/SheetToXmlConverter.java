package com.capstone.sheet.service;

import com.capstone.exception.InternalServerException;
import com.capstone.sheet.dto.SheetCreateMeta;
import lombok.extern.slf4j.Slf4j;
import org.apache.tomcat.util.http.fileupload.FileUtils;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class SheetToXmlConverter {
    /**
     * write sheetFile's content on target file
     * @param target file to write sheetFile's content
     * @param sheetFile multipart file about sheet content
     * @throws IOException if failed to write file
    * */
    private void writeFile(Path target, MultipartFile sheetFile) throws IOException {
        try(OutputStream fos = Files.newOutputStream(target)) {
            fos.write(sheetFile.getBytes());
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
                "docker", "run", "--rm",
                "-v", String.format("%s:/input", inputPath),
                "-v", String.format("%s:/output", outputPath),
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
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
        builder.command(commandBuilder(inputPath, outputPath));
        String outputFileName = "output.xml";
        Path outputFile;
        try{
            Process process = builder.start();
            consumeStream(process.getInputStream());  // stdout
            consumeStream(process.getErrorStream());  // stderr
            process.waitFor(3, TimeUnit.MINUTES);
            outputFile = Paths.get(outputPath, outputFileName);
            return Files.readAllBytes(outputFile);
        }catch (IOException | InterruptedException e){
            String errorMessage = "SheetToXmlConverter.convert : " + e.getMessage();
            e.printStackTrace();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        }
    }

    /**
     * Score file life cycle management and xml conversion functions
     * @param sheetCreateMeta metadata of sheet
     * @param sheetFile sheet file data
     * @return byte array of sheet xml
    * */
    public byte[] convertToXml(SheetCreateMeta sheetCreateMeta, MultipartFile sheetFile) {
        Path target;
        String tmpDir = System.getProperty("java.io.tmpdir");
        Path sheetDir = Paths.get(tmpDir,"sheet", sheetCreateMeta.getUserEmail());
        try {
            Files.createDirectories(sheetDir);
            target = Files.createTempFile(
                    sheetDir,
                    "sheet_",
                    "."+ sheetCreateMeta.getFileExtension());
            writeFile(target, sheetFile);
            String inputPath = sheetDir.toString();
            String outputPath = Paths.get(sheetDir.toString(), "output").toString();
            return processConvert(inputPath, outputPath);
        }catch (IOException e){
            String errorMessage = "SheetToXmlConverter.convertToXml : " + e.getMessage();
            log.error(errorMessage);
            throw new InternalServerException(errorMessage);
        }finally {
            deleteDirectory(sheetDir);
        }
    }
}
