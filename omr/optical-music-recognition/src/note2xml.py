from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString


class Note2XML:
    @staticmethod
    def create_musicxml(note_data, save_path):
        notes_data = note_data["notes"]
        attributes_data = note_data["attributes"]

        # Create root element
        score_partwise = Element("score-partwise", {"version": "4.0"})

        # Create part-list and score-part elements
        part_list = SubElement(score_partwise, "part-list")
        score_part = SubElement(part_list, "score-part", {"id": "P1"})
        part_name = SubElement(score_part, "part-name")
        part_name.text = "Track 1"

        # Create part element
        part = SubElement(score_partwise, "part", {"id": "P1"})

        # Create measure element
        measure = SubElement(part, "measure", {"number": "1"})
        attributes = SubElement(measure, "attributes")
        # Add attributes to measure
        divisions = SubElement(attributes, "divisions")
        divisions.text = str(attributes_data["divisions"])

        time = SubElement(attributes, "time")
        beats = SubElement(time, "beats")
        beats.text = str(attributes_data["beats"])
        beat_type = SubElement(time, "beat-type")
        beat_type.text = str(attributes_data["beat-type"])

        clef = SubElement(attributes, "clef")
        sign = SubElement(clef, "sign")
        sign.text = "percussion"

        # Loop through note data and create note elements
        for note_info in notes_data:
            note = SubElement(measure, "note")

            if "chord" in note_info and note_info["chord"]:
                chord = SubElement(note, "chord")

            if not ("step" in note_info and "octave" in note_info):
                rest = SubElement(note, "rest", {"measure": "yes"})

            if "step" in note_info and "octave" in note_info:
                unpitched = SubElement(note, "unpitched")

                display_step = SubElement(unpitched, "display-step")
                step = note_info["step"]
                display_step.text = step

                display_octave = SubElement(unpitched, "display-octave")
                octave = str(note_info["octave"])
                display_octave.text = octave

                stem = SubElement(note, "stem")
                stem.text = "up"

                # G5, A5, B5: x notehead
                pitch = step + octave
                if pitch in ["G5", "A5", "B5"]:
                    notehead = SubElement(note, "notehead")
                    notehead.text = "x"

            if "duration" in note_info and "type" in note_info:
                duration = SubElement(note, "duration")
                duration.text = str(note_info["duration"])

                note_type = SubElement(note, "type")
                note_type.text = note_info["type"]

        # XML을 파일로 저장
        tree = parseString(tostring(score_partwise))
        with open(save_path, "wb") as f:
            f.write(tree.toprettyxml(encoding="UTF-8"))


"""
<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE score-partwise>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Track 1</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>32</divisions>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>percussion</sign>
        </clef>
      </attributes>
      
      note...
      
    </measure>
  </part>
</score-partwise>
"""

"""
<note>
  <unpitched>
    <display-step>F</display-step>
    <display-octave>4</display-octave>
  </unpitched>
  <duration>32</duration>
  <type>quarter</type>
  <stem>up</stem>
</note>
<note>
  <chord/>
  <unpitched>
    <display-step>G</display-step>
    <display-octave>5</display-octave>
  </unpitched>
  <duration>32</duration>
  <type>quarter</type>
  <stem>up</stem>
  <notehead>x</notehead>
</note>
"""

"""
<note>
  <rest measure="yes"/>
  <duration>32</duration>
  <type>quarter</type>
</note>
"""
