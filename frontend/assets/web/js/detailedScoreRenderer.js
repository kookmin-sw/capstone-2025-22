import { OpenSheetMusicDisplay } from "./opensheetmusicdisplay.min.js";

export async function renderDetailedScore(xmlText, practiceInfo, options = {}) {
  const container = document.getElementById("detailedOsmd");
  container.innerHTML = "";

  // 1) OSMD 인스턴스 생성 + EngravingRules 설정
  const osmd = new OpenSheetMusicDisplay(container, {
    backend: "svg",
    autoResize: true,
    drawTitle: false,
    drawComposer: false,
    drawPartNames: false,
    drawingParameters: "compact",
    drawMeasureNumbers: true,
    pageBackgroundColor: "#FFFFFF",
    renderSingleHorizontalStaffline: false,
    });

  // 한 시스템(줄)에 4마디씩
  osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = 4;
  osmd.EngravingRules.FixSystemDistance = true;

  // 2) XML 로드 & 렌더
  await osmd.load(xmlText);
  await osmd.render();

  // 3) 전체 노트헤드 기본 색상(검정)으로 초기화
  const allNoteheads = container.querySelectorAll(".vf-notehead");
  allNoteheads.forEach(nh => nh.setAttribute("fill", options.colorDefault || "#000000"));

  // 4) 마디별 1차 채점 정보 적용: 틀린 음표만 회색
  practiceInfo.forEach(measureResult => {
    const mNum = parseInt(measureResult.measureNumber, 10);
    const beatResults = measureResult.beatScoringResults;  // [true,false,true,...]
    const measureIdx = mNum - 1;

    // OSMD 내부에서 그래픽 요소에 접근
    const measure = osmd.GraphicSheet.getMeasureElement(measureIdx);
    if (!measure) return;

    console.log(
    `measure ${mNum}`, 
    "DOM noteheads:", noteheads.length, 
    "Dart cursors:", practiceInfo[measureIdx].beatScoringResults.length
    );

    // 해당 마디의 notehead SVG 노드만 골라서
    const noteheads = measure.querySelectorAll(".vf-notehead");
    noteheads.forEach((nh, idx) => {
      if (beatResults[idx] === false) {
        nh.setAttribute("fill", options.colorWrong1 || "#888888");
      }
      // true 면 기본 검정(#000000) 유지
    });
  });

}
