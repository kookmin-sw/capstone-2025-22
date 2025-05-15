// 커서 스타일
const cursorWidth = 20;
const cursorHeight = 100;

// 전역 변수 정의
let xmlData = "";
let BPM = 60; // 기본 BPM 값
let beatsPerMeasure = 4; // 박자 값 (4/4인 경우 4, 3/4인 경우 3...)
let isRendered = false; // 렌더링 완료 여부 체크 변수

const defaultOptions = {
  autoResize: true,
  backend: "canvas",
  defaultColorNotehead: "#000000",
  defaultColorStem: "#000000",
  drawTitle: false,
  drawComposer: false,
  drawClef: false,
  drawTimeSignature: false,
  drawPartNames: false,
  drawingParameters: "compact",
  drawMeasureNumbers: true,
  pageBackgroundColor: "transparent",
  renderSingleHorizontalStaffline: false,
};

// 전체 캔버스에서 한 번만 이미지 생성
async function createSheetImage(fullCanvas) {
  return fullCanvas.toDataURL("image/png").split(",")[1].trim();
}

function getScale(osmd, fullCanvas) {
  // 1️⃣ unit(px) ← OSMD 내부 단위(1 space) ↔ 픽셀 변환 계수
  let unitPx =
        osmd.drawer?.unitInPixels // (구버전 dev-build)
     ?? (typeof opensheetmusicdisplay !== 'undefined' // (정식 번들)
         ? opensheetmusicdisplay.unitInPixels
         : undefined);
  if (unitPx == null) unitPx = 10; // 🔙 최후의 안전값

  // 2️⃣ 확대 / 축소 배율
  const zoom = osmd.Zoom || 1;

  // 3️⃣ Hi-DPI(레티나) 캔버스 → CSS 픽셀 보정
  const cssFactor = fullCanvas.width /
                    fullCanvas.getBoundingClientRect().width;

  return unitPx * zoom * cssFactor; // 최종 스케일
}

// 시스템(줄) 단위 PNG 자르기 + 줄별 bbox(절대좌표)까지 반환
async function cropLineImages(fullCanvas, osmd) {
  // 모든 페이지의 MusicSystem 을 1차원 배열로
  const systems = osmd.GraphicSheet.MusicPages.flatMap(p => p.MusicSystems);
  const images = [];
  const bounds  = []; // 줄별 [top, bot] 절대좌표 저장

  // 스케일 계산
  const scale = getScale(osmd, fullCanvas);
  const cssFactor = fullCanvas.width / fullCanvas.getBoundingClientRect().width;

  // 첫 시스템의 절대 Y → 잘라낼 때의 기준점(offset)
  const topOffset = systems[0].PositionAndShape.AbsolutePosition.y * scale;

  // ─── 시스템을 순회하면서 한 줄씩 자르기 ───
  for (let i = 0; i < systems.length; i++) {
    const system = systems[i];
    const sysPosY  = system.PositionAndShape.AbsolutePosition.y * scale;
    const sysBotY  = sysPosY + system.PositionAndShape.Size.height * scale;

    let minY = Infinity; // 가장 높은(위) 음표
    let maxY = -Infinity; // 가장 낮은(아래) 음표
    let noteCount = 0;

    // skyline / bottomline 기반 음표 영역 탐색
    for (const gm of system.GraphicalMeasures.flat()) {
      for (const se of (gm.staffEntries ?? [])) {
        const seAbsY = se.PositionAndShape.AbsolutePosition.y * scale;

        // staffEntry 내부 상대좌표 → 절대좌표 변환
        const top = seAbsY + se.getSkylineMin() * scale;
        const bottom = seAbsY + se.getBottomlineMax() * scale;
        if (Number.isFinite(top) && Number.isFinite(bottom)) {
          minY = Math.min(minY, top);
          maxY = Math.max(maxY, bottom);
          noteCount++; // 하나라도 잡히면 '음표 있음' 표시
        }
      }
    }

    // 음표 bbox 와 시스템 bbox 중 더 넓은 쪽을 자르기 범위로
    const cropTop = noteCount ? Math.min(minY, sysPosY) : sysPosY;
    const cropBot = noteCount ? Math.max(maxY, sysBotY) : sysBotY;

    const y = Math.round(cropTop - topOffset); // 캔버스에서의 시작 Y
    const height = Math.round(cropBot - cropTop);
    
    const sysPosX = system.PositionAndShape.AbsolutePosition.x * scale;
    const sysWidth = system.PositionAndShape.Size.width * scale;

    console.log(`[crop] line ${i}: y=${y}, h=${height}, canvasH=${fullCanvas.height}`);

    // 캔버스 바깥 범위 방지
    if (height <= 0 || y + height > fullCanvas.height) {
      console.warn(`⚠️ skip line ${i} (out of canvas)`);
      continue;
    }

    // 잘라서 임시 캔버스에 복사 → Base64 PNG
    const tmp = document.createElement('canvas');
    tmp.width = Math.round(sysWidth);  tmp.height = height;
    tmp.getContext('2d').drawImage(
      fullCanvas, Math.round(sysPosX), y, Math.round(sysWidth), height,
      0, 0, Math.round(sysWidth), height
    );
    images.push(tmp.toDataURL('image/png').split(',')[1]);
    // 절대좌표 -> 이미지 내부 좌표로 변경
    bounds.push({
      left:   sysPosX   / cssFactor,  // 잘라낸 시스템의 시작 위치 (CSS px)
      width:  sysWidth  / cssFactor,  // 잘라낸 시스템의 폭 (CSS px)
      top: (cropTop - topOffset) / cssFactor,
      bot: (cropBot - topOffset) / cssFactor
    });
    
    // 2줄마다 한 번씩 이벤트루프 양보 (UI 프리징 방지)
    if (i % 2 === 1) await new Promise(requestAnimationFrame); 
  }
  return { images, bounds };
}

// BPM 파싱하는 함수
function extractBPMFromXML(xmlText) {
  const match = xmlText.match(/<sound[^>]*tempo=\"([\d.]+)\"/);
  if (match) {
    BPM = parseFloat(match[1]);
  } else {
    console.warn("BPM(tempo) 태그를 찾지 못했습니다.");
  }
}

// time 서명(박자표) 파싱하는 함수
function extractTimeSignatureFromXML(xmlText) {
  // <time>…<beats>X</beats>…</time> 중 X를 꺼냄
  const match = xmlText.match(/<time>[\s\S]*?<beats>(\d+)<\/beats>/);
  if (match) {
    beatsPerMeasure = parseInt(match[1], 10);
  } else {
    console.warn("Time signature (<beats>) 태그를 찾지 못했습니다.");
  }
}

function getCursorList(osmdCursor, lineBounds) {
  // 1) 실제 음표(ts)만 모은 rawCursorList 생성
  const rawCursorList = [];
  osmdCursor.reset();
  osmdCursor.show();
  try {
    while (!osmdCursor.iterator.endReached) {
      // 실제 음표(VoiceEntry)가 있는 틱만 필터
      const visible = osmdCursor.iterator.CurrentVisibleVoiceEntries();
      if (visible.length > 0) {
        rawCursorList.push( getCursorInfo(osmdCursor, lineBounds) );
      }
      osmdCursor.next();
    }
  } catch (e) {
    console.error("❗ 커서 위치 수집 중 오류 발생", e);
  }
  osmdCursor.hide();
  console.log(`📊 Raw cursors: ${rawCursorList.length}`);

  // 2) rawCursorList → fullCursorList: 휴지(rest) 구간을 1-beat 단위로 채움
  const fullCursorList = [];
  for (let i = 0; i < rawCursorList.length - 1; i++) {
    const curr = rawCursorList[i];
    const next = rawCursorList[i + 1];

    // 2-1) 실제 음표 포인트
    fullCursorList.push(curr);

    // 2-2) 마디 사이 휴지 구간
    //    next.ts와 curr.ts 사이가 1 beat 이상이면,
    //    Math.ceil(curr.ts)+1  부터 Math.floor(next.ts) 까지 1씩 추가
    const startBeat = Math.ceil(curr.ts);
    const endBeat   = Math.floor(next.ts);
    for (let beat = startBeat; beat <= endBeat; beat++) {
      // beat 만큼 ts를 옮긴 새로운 포인트 생성
      fullCursorList.push({
        ...curr,
        ts: beat,
      });
    }
  }
  // 3) 마지막 실제 음표 포인트도 추가
  if (rawCursorList.length > 0) {
    fullCursorList.push(rawCursorList[rawCursorList.length - 1]);
  }
   // 4) 마지막 마디 끝(ts = (마지막 마디 인덱스 + 1) * beatsPerMeasure)까지
   if (rawCursorList.length > 0) {
    // rawCursorList 마지막 항목에서 measureNumber 가져오기 (0-based)
    const lastRaw = rawCursorList[rawCursorList.length - 1];
    const totalBeats = (lastRaw.measureNumber + 1) * beatsPerMeasure; // ex. 16 * 4 = 64
    // fullCursorList 맨 끝 포인트
    const lastFull = fullCursorList[fullCursorList.length - 1];
    // lastFull.ts 다음 정수부터 totalBeats까지 반복
    const startBeat = Math.floor(lastFull.ts) + 1;
    for (let beat = startBeat; beat <= totalBeats; beat++) {
      fullCursorList.push({
        ...lastFull,
        ts: beat,
      });
    }
  }

  console.log(`📊 Full cursors: ${fullCursorList.length}`);
  return { rawCursorList, fullCursorList };
}

// 커서 위치·크기·타임스탬프 한꺼번에 계산
function getCursorInfo(osmdCursor, lineBounds) {
  const fullCanvas = window._osmdFullCanvas;
  const canvasRect = fullCanvas.getBoundingClientRect();
  const elRect = osmdCursor.cursorElement.getBoundingClientRect();
  const cssFactor = fullCanvas.width / canvasRect.width;

  // ② CSS logical px → 캔버스 실제 px
  const xCss = (elRect.left - canvasRect.left) + cursorWidth/2;
  const yCss = (elRect.top  - canvasRect.top ) - cursorHeight/2;
  const x    = xCss * cssFactor;
  const y    = yCss * cssFactor;

  const measureNumber = osmdCursor.iterator.CurrentMeasureIndex || 0;
  
  // OSMD는 16분음표(0.25)를 기준으로 타임스탬프를 계산하므로
  const ts = osmdCursor.iterator.currentTimeStamp.realValue * 4;

  const measuresPerLine = osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem || 4;
  const lineIndex = Math.floor(measureNumber / measuresPerLine);

  const { left: lineLeft = 0, width: lineWidth = fullCanvas.width } =
    lineBounds[lineIndex] || {};

  // x(캔버스 전체 px) → 잘라낸 이미지(시스템) 내부 위치 → 비율
  const xRatio = (x / cssFactor - lineLeft) / lineWidth;

  // 1) lineBounds 에서 이 커서가 속한 줄의 CSS logical 픽셀 경계 꺼내기
  //    lineBounds 배열에는 { top: cssPx, bot: cssPx } 형태로 저장되어 있습니다.
  const { top: cropTopCss = 0, bot: cropBotCss = canvasRect.height } =
  lineBounds[lineIndex] || {};

  // 2) CSS logical 좌표(yCss) 기준, 줄 안에서의 상대 위치 비율 계산
  //    (cropBotCss - cropTopCss) 이 0 이 아닌 경우만 계산
  const yRatio =
  cropBotCss > cropTopCss
    ? (yCss - cropTopCss) / (cropBotCss - cropTopCss)
    : 0;

    
  return { x, y, w: cursorWidth, h: cursorHeight, ts, xRatio, yRatio, measureNumber, lineIndex };
}

window.startOSMDFromFlutter = async function () {
  if (isRendered) {
    return;
  }
  // MusicXML 불러오기 (Flutter → JS)
  const xmlText = await window.flutter_inappwebview.callHandler("sendFileToOSMD");
  xmlData = xmlText; // 전역 저장

  // BPM, 박자 추출 
  extractBPMFromXML(xmlText);
  extractTimeSignatureFromXML(xmlText);

  // OSMD 인스턴스 초기화
  const container = document.getElementById("osmdCanvas");
  container.innerHTML = ""; // 기존 것 초기화

  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    ...defaultOptions,
    drawFromMeasureNumber: 1,
    drawUpToMeasureNumber: Number.MAX_SAFE_INTEGER,
  });

  osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = 4;
  osmd.EngravingRules.FixSystemDistance = true;

  await osmd.load(xmlText, "");
  window.osmd = osmd;
  await osmd.render();
  osmd.cursor.reset()
  osmd.cursor.hide();

  // 줄별 뷰(캔버스) 보이기
  document.getElementById('osmdCanvas').style.display     = 'block';
  // 상세 뷰(SVG) 숨기기
  document.getElementById('detailedOsmd').style.display   = 'none';


  await new Promise(resolve => setTimeout(resolve, 100)); // JS 이벤트 루프 돌기
  await new Promise(requestAnimationFrame);

  const fullCanvas = container.querySelector("#osmdCanvasVexFlowBackendCanvas1");
  window._osmdFullCanvas = fullCanvas;

  // 전체 악보 이미지 
  const sheetImage = await createSheetImage(fullCanvas);

  // ▸ 줄별 PNG + bbox 얻기
  const { images: lineImages, bounds: lineBounds } = await cropLineImages(fullCanvas, osmd);
  
  const { rawCursorList, fullCursorList } = getCursorList(osmd.cursor, lineBounds);
  console.log("📊 rawCursorList ts 리스트:", rawCursorList.map(c => c.ts));
  
  // 악보 줄 수: MusicXML의 <measure> 태그 개수를 세어서 4마디씩 나눈 줄 수로 계산
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
  const measures = xmlDoc.getElementsByTagName('measure');
  const measuresPerLine = osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem || 4;
  const systemCount = Math.ceil(measures.length / measuresPerLine);

  // XML 데이터 (Flutter에서 받아온 원본)
  const xmlBase64 = btoa(unescape(encodeURIComponent(xmlText)));

  window.flutter_inappwebview.callHandler("getDataFromOSMD", 
    sheetImage,
    {
      cursorList: fullCursorList,
      rawCursorList,
      bpm: BPM,
      canvasWidth: fullCanvas.width,
      canvasHeight: fullCanvas.height,
      xmlData: xmlBase64,
      lineImages,
      lineBounds,
      lineCount: systemCount,
      totalMeasures: measures.length // 악보 총 마디 개수
    }
  );
  isRendered = true;
};