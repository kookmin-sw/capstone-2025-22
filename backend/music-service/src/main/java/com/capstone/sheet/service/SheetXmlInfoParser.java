package com.capstone.sheet.service;

import com.capstone.sheet.constants.MusicXmlTags;
import com.capstone.sheet.constants.MusicXmlTextValues;
import com.capstone.sheet.dto.musicXml.*;
import com.capstone.sheet.constants.MusicXmlAttributes;
import com.capstone.sheet.utils.FieldLister;
import org.springframework.stereotype.Component;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Component
public class SheetXmlInfoParser {

    private final DocumentBuilder documentBuilder;

    public SheetXmlInfoParser() throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
        factory.setFeature("http://xml.org/sax/features/validation", false);
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", false);
        try {
            this.documentBuilder = factory.newDocumentBuilder();
        } catch (Exception e) {
            throw new RuntimeException("Error creating DocumentBuilder", e);
        }
    }

    public Optional<Element> getFirstElementByTagName(Element element, String tagName){
        NodeList nodeList = element.getElementsByTagName(tagName);
        if(nodeList.getLength() > 0) return Optional.of((Element) nodeList.item(0));
        return Optional.empty();
    }

    public Optional<String> getFirstElementTextContentByTagName(Element element, String tagName){
        Element firstElement = getFirstElementByTagName(element, tagName).orElse(null);
        if(firstElement != null) return Optional.of(firstElement.getTextContent());
        return Optional.empty();
    }

    public Optional<Element> getPitchTypeElement(Element noteElement){
        List<String> noteTypes = FieldLister.getFieldValues(MusicXmlTags.Pitch.class);
        for(String noteType : noteTypes){
            if(noteElement.getElementsByTagName(noteType).getLength() > 0){
                return getFirstElementByTagName(noteElement, noteType);
            }
        }
        return Optional.empty();
    }

    public String getNoteHead(Element noteElement){
        List<String> noteHeads = FieldLister.getFieldValues(MusicXmlTextValues.NoteHead.class);
        Element noteHeadElement = getFirstElementByTagName(noteElement, MusicXmlTags.NOTE_HEAD).orElse(null);
        if(noteHeadElement == null) return MusicXmlTextValues.NoteHead.NORMAL;
        for(String noteHead : noteHeads){
            if(noteHeadElement.getTextContent().equals(noteHead)) return noteHead;
        }
        return MusicXmlTextValues.NoteHead.NORMAL;
    }

    public String getNoteType(Element noteElement){
        List<String> notTypes = FieldLister.getFieldValues(MusicXmlTextValues.NoteType.class);
        Element noteTypeElement = getFirstElementByTagName(noteElement, MusicXmlTags.TYPE).orElse(null);
        if(noteTypeElement != null){
            for(String noteType : notTypes){
                if(noteTypeElement.getTextContent().equals(noteType)) return noteType;
            }
        }
        return null;
    }

    public PitchInfo resolveNote(Element noteElement){
        String defaultX = noteElement.getAttribute(MusicXmlAttributes.DEFAULT_X);
        String noteHead = getNoteHead(noteElement);
        String noteType = getNoteType(noteElement);
        String duration = getFirstElementTextContentByTagName(noteElement, MusicXmlTags.DURATION).orElse(null);
        Optional<Element> noteTypeElement = getPitchTypeElement(noteElement);
        String pitchType = noteTypeElement.map(Element::getTagName)
                .orElse(null);
        String displayStep = noteTypeElement
                .flatMap(el -> getFirstElementTextContentByTagName(el, MusicXmlTags.DisplayInfo.DISPLAY_STEP))
                .orElse(null);
        String displayOctave = noteTypeElement
                .flatMap(el -> getFirstElementTextContentByTagName(el, MusicXmlTags.DisplayInfo.DISPLAY_OCTAVE))
                .orElse(null);
        return PitchInfo.builder()
                .defaultX(defaultX)
                .noteHead(noteHead)
                .noteType(noteType)
                .pitchType(pitchType)
                .duration(duration)
                .displayStep(displayStep)
                .displayOctave(displayOctave)
                .build();
    }

    public boolean isChord(Element noteElement){
        return noteElement.getElementsByTagName(MusicXmlTags.CHORD).getLength() > 0;
    }

    public MeasureInfo resolveMeasure(Element measureElement, int divisions){
        String measureNumber = measureElement.getAttribute(MusicXmlAttributes.MEASURE_NUMBER);
        NodeList notes = measureElement.getElementsByTagName(MusicXmlTags.NOTE);
        List<NoteInfo> noteList = new ArrayList<>();
        double cumulatedOnset = 0.0;
        for(int j=0; j<notes.getLength(); j++){
            Element noteElement = (Element) notes.item(j);
            PitchInfo pitchInfo = resolveNote(noteElement);
            if(isChord(noteElement) && !noteList.isEmpty()){
                noteList.get(noteList.size()-1).getPitchList().add(pitchInfo);
            }else{
                double beatDuration = Double.parseDouble(pitchInfo.getDuration()) / divisions;
                double startOnset = cumulatedOnset;
                double endOnset = cumulatedOnset + beatDuration;
                NoteInfo newNoteInfo = NoteInfo.builder()
                        .pitchList(new ArrayList<>(List.of(pitchInfo)))
                        .startOnset(startOnset)
                        .endOnset(endOnset).build();
                noteList.add(newNoteInfo);
                cumulatedOnset = endOnset;
            }
        }
        return MeasureInfo.builder()
                .measureNumber(measureNumber)
                .noteList(noteList)
                .build();
    }

    public Optional<MusicXmlMetaData> resolveMusicXmlMetaData(Element firstMeasure){
        Optional<Element> attributes = getFirstElementByTagName(firstMeasure, MusicXmlTags.ATTRIBUTES);
        return attributes.map(attr -> {
            MusicXmlMetaData metaData = new MusicXmlMetaData();
            getFirstElementTextContentByTagName(attr, MusicXmlTags.DIVISIONS)
                    .ifPresent(division -> metaData.setDivision(Integer.parseInt(division)));
            getFirstElementByTagName(attr, MusicXmlTags.TIME).ifPresent(timeElement -> {
               getFirstElementTextContentByTagName(timeElement, MusicXmlTags.BEATS)
                       .ifPresent(beats -> metaData.setBeat(Integer.parseInt(beats)));
               getFirstElementTextContentByTagName(timeElement, MusicXmlTags.BEAT_TYPE)
                       .ifPresent(beatType -> metaData.setBeatType(Integer.parseInt(beatType)));
            });
            return metaData;
        });
    }

    public List<PartInfo> parseXmlInfo(byte[] sheetInfo) throws Exception {
        List<PartInfo> partInfoList = new ArrayList<>();
        InputStream inputStream = new ByteArrayInputStream(sheetInfo);
        Document document = documentBuilder.parse(inputStream);
        NodeList partNodes = document.getElementsByTagName(MusicXmlTags.PART);
        for (int i = 0; i < partNodes.getLength(); i++) {
            Element partElement = (Element) partNodes.item(i);
            NodeList measures = partElement.getElementsByTagName(MusicXmlTags.MEASURE);
            // set default part info
            PartInfo.PartInfoBuilder partInfoBuilder = PartInfo.builder();
            ArrayList<MeasureInfo> measureInfoList = new ArrayList<>();
            // set meta data if exists
            resolveMusicXmlMetaData((Element) measures.item(0)).ifPresent(meta -> partInfoBuilder
                    .beats(meta.getBeat())
                    .beatType(meta.getBeatType())
                    .divisions(meta.getDivision()));
            // parse measures and add to part info list
            int divisions = partInfoBuilder.build().getDivisions();
            for (int j = 0; j < measures.getLength(); j++) {
                Element measureElement = (Element) measures.item(j);
                MeasureInfo measure = resolveMeasure(measureElement, divisions);
                measureInfoList.add(measure);
            }
            // add part info to partInfoList
            partInfoBuilder.measureList(measureInfoList);
            partInfoList.add(partInfoBuilder.build());
        }
        return partInfoList;
    }
}
