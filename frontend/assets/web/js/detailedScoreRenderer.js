window.renderDetailedScore = async function(xmlText, practiceInfo, options = {}) {
  const container = document.getElementById("detailedOsmd");
  container.innerHTML = "";

  // 상세 뷰(SVG) 보이기
  document.getElementById('detailedOsmd').style.display   = 'block';
  // 줄별 뷰(캔버스) 숨기기
  document.getElementById('osmdCanvas').style.display     = 'none';

  // OpenSheetMusicDisplay 전역에서 꺼내 쓰기
  const OpenSheetMusicDisplay = window.opensheetmusicdisplay.OpenSheetMusicDisplay;

  // 1) OSMD 인스턴스 생성 + EngravingRules 설정
  const osmd = new OpenSheetMusicDisplay(container, {
    backend: "svg",
    autoResize: false,
    drawTitle: false,
    drawComposer: false,
    drawPartNames: false,
    drawingParameters: "compact",
    drawMeasureNumbers: true,
    pageBackgroundColor: "#d97d6c",
    renderSingleHorizontalStaffline: false,
    });

  // 한 시스템(줄)에 4마디씩
  osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = 4;
  osmd.EngravingRules.FixSystemDistance = true;

  // 2) XML 로드 & 렌더
  await osmd.load(xmlText);
  await osmd.render();
  await new Promise(requestAnimationFrame);

  const allMeasures = osmd.GraphicSheet.MeasureList.flat();
  
  // 1) measureNumber 순서대로 정렬
  practiceInfo
    .sort((a, b) => parseInt(a.measureNumber, 10) - parseInt(b.measureNumber, 10))
    .forEach(result => {
      const mIdx = parseInt(result.measureNumber, 10) - 1;
      const measure = allMeasures[mIdx];
      if (!measure) {
        console.warn(`❗ no measure for #${result.measureNumber}`);
        return;
      }

      const fr = result.beatScoringResults  || [];
      const sr = result.finalScoringResults || [];

      // 1) 이 마디 안의 모든 GraphicalNote를 timestamp별로 그룹핑
      const notesByTime = measure.staffEntries
        .flatMap(se => se.graphicalVoiceEntries)
        .flatMap(ve => ve.notes)
        .reduce((map, gNote) => {
          // sourceNote.parentStaffEntry.AbsoluteTimestamp.RealValue 에 접근
          const t = gNote
            .sourceNote
            .parentStaffEntry
            .AbsoluteTimestamp
            .RealValue;
          if (!map.has(t)) map.set(t, []);
          map.get(t).push(gNote);
          return map;
        }, new Map());

      // 2) 타임스탬프별로 묶고 시간순으로 정렬, 한 타임스탬프당 하나의 대표 노트
      const timeEntries = Array.from(notesByTime.entries())
        .sort((a, b) => a[0] - b[0]);    // [ [timestamp, [gNote, gNote...]], ... ]

      // 대표 노트 리스트 (그룹당 첫번째 노트만)
      const notesInMeasure = timeEntries.map(([t, group]) => group[0]);

      console.log(
        `▶ measure #${result.measureNumber}: FR=${fr.length}, SR=${sr.length}, actualNotes=${notesInMeasure.length}`
      );
      // 3) 순서대로 색 덮어쓰기
      notesInMeasure.forEach((repNote, i) => {
        let color = options.colorDefault || "#000";
        if (sr.length === 0) {
          if (fr[i] === false) color = options.colorWrong1;
        } else {
          const w1 = fr[i] === false;
          const w2 = sr[i] === false;
          if      (w1 && w2) color = options.colorBothWrong;
          else if (w1)       color = options.colorWrong1Only;
          else if (w2)       color = options.colorWrong2Only;
        }

        // **그룹 전체**(둥근 음표 + X) 에 동일한 색 입히기
        const group = timeEntries[i][1];  // 같은 t 의 모든 gNote 들
        group.forEach(gNote => {
          const svgG = gNote.getSVGGElement();
          if (!svgG) return;
          svgG.querySelectorAll('[fill], [stroke]').forEach(el => {
            if (el.hasAttribute('fill'))   el.setAttribute('fill',   color);
            if (el.hasAttribute('stroke')) el.setAttribute('stroke', color);
          });
        });
      });
  });
};