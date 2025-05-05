// 커서 스타일
const cursorWidth = 20;
const cursorHeight = 100;

// 전역 변수 정의
let xmlData = "";
let BPM = 60; // 기본 BPM 값
let isRendered = false; // 렌더링 완료 여부 체크 변수

const defaultOptions = {
  autoResize: false,
  backend: "canvas",
  defaultColorNotehead: "#000000",
  defaultColorStem: "#000000",
  drawTitle: false,
  drawComposer: false,
  drawClef: false,
  drawTimeSignature: false,
  drawPartNames: false,
  drawingParameters: "default",
  drawMeasureNumbers: true,
  pageBackgroundColor: "#FFFFFF",
  renderSingleHorizontalStaffline: false,
};

// 전체 캔버스에서 한 번만 이미지 생성
async function createSheetImage(fullCanvas) {
  return fullCanvas.toDataURL("image/png").split(",")[1].trim();
}

async function cropLineImages(fullCanvas) {
  const pages = window.osmd?.GraphicSheet?.MusicPages || [];
  if (pages.length === 0) {
    console.error("❗ MusicPages를 찾을 수 없습니다.");
    return [];
  }

  const systems = pages.flatMap(p => p.MusicSystems);
  console.log("✅ 전체 시스템 수:", systems.length);

  const MARGIN = 10;
  const lineImages = [];

  for (let i = 0; i < systems.length; i++) {
    const shape = systems[i].PositionAndShape;
    const y = Math.max((shape.AbsolutePosition.y) - MARGIN, 0);
    const nextSystem = systems[i + 1];
    const nextY = nextSystem
      ? (nextSystem.PositionAndShape.AbsolutePosition.y)
      : fullCanvas.height;
  
    const h = Math.min(Math.ceil(nextY - y), fullCanvas.height - y);
    const x = 0;
    const w = fullCanvas.width;

    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    off.getContext("2d").drawImage(fullCanvas, x, y, w, h, 0, 0, w, h);

    const base64 = off.toDataURL("image/png").split(',')[1].trim();
    console.log(`📏 crop region[${i}]: x=${x}, y=${y}, w=${w}, h=${h}, len=${base64.length}`);
    
    lineImages.push(base64);
  }

  return lineImages;
}

function extractBPMFromXML(xmlText) {
  const match = xmlText.match(/<sound[^>]*tempo=\"([\d.]+)\"/);
  if (match) {
    BPM = parseFloat(match[1]);
  } else {
    console.warn("BPM(tempo) 태그를 찾지 못했습니다.");
  }
}

function getCursorList(osmdCursor) {
  const cursorList = [];
  osmdCursor.show();
  try {
    while (!osmdCursor.iterator.endReached) {
      // 직접 박자 정보(ts)와 커서 Element 위치를 수집
      cursorList.push(getCursorInfo(osmdCursor));
      osmdCursor.next();              
    }
  } catch (e) {
    console.error("❗ 커서 위치 수집 중 오류 발생", e);
  }
  osmdCursor.hide();
  return cursorList;
}

// 커서 위치·크기·타임스탬프 한꺼번에 계산
function getCursorInfo(osmdCursor) {
  const fullCanvas = window._osmdFullCanvas;
  const canvasRect = fullCanvas.getBoundingClientRect();
  const elRect = osmdCursor.cursorElement.getBoundingClientRect();

  // CSS 상의 logical px → 캔버스 기준 상대 좌표
  const x = elRect.left - canvasRect.left + (cursorWidth / 2);
  const y = elRect.top - canvasRect.top - (cursorHeight / 2);
  const ts = osmdCursor.iterator.currentTimeStamp.realValue;
  const xRatio = x / canvasRect.width;

  return { x, y, w: cursorWidth, h: cursorHeight, ts, xRatio };
}

window.startOSMDFromFlutter = async function () {
  if (isRendered) {
    return;
  }

  // MusicXML 불러오기 (Flutter → JS)
  const xmlText = await window.flutter_inappwebview.callHandler("sendFileToOSMD");
  xmlData = xmlText; // 전역 저장

  // BPM 추출 
  extractBPMFromXML(xmlText);

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

  await new Promise(resolve => setTimeout(resolve, 100));
  await new Promise(requestAnimationFrame);
  await new Promise(requestAnimationFrame);
  console.log("📏 전체 마디 수:", osmd.Sheet.SourceMeasures.length);

  const musicSystems = osmd.GraphicSheet?.MusicPages?.[0]?.MusicSystems || [];
  console.log(`🧱 전체 시스템 개수: ${musicSystems.length}`);

  musicSystems.forEach((sys, i) => {
    const shape = sys.PositionAndShape;
    const absX = shape.AbsolutePosition.x.toFixed(2);
    const absY = shape.AbsolutePosition.y.toFixed(2);
    const w = shape.Size.width.toFixed(2);
    const h = shape.Size.height.toFixed(2);
    console.log(`📌 system[${i}] - absX: ${absX}, absY: ${absY}, width: ${w}, height: ${h}`);
  });

  const fullCanvas = container.querySelector("#osmdCanvasVexFlowBackendCanvas1");
  console.log("🖼️ fullCanvas size:", fullCanvas?.width, fullCanvas?.height);

  window._osmdFullCanvas = fullCanvas;
  const sheetImage = await createSheetImage(fullCanvas);
  console.log("🖼️ sheetImage Base64 length:", sheetImage.length);

  // 악보 줄별 이미지 생성
  const lineImages = await cropLineImages(fullCanvas);

  // 커서 리스트 수집
  const cursorList = getCursorList(osmd.cursor);

  // 악보 줄 수: MusicXML의 <measure> 태그 개수를 세어서 4마디씩 나눈 줄 수로 계산
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
  const measures = xmlDoc.getElementsByTagName('measure');
  const measuresPerLine = osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem || 4;
  const systemCount = Math.ceil(measures.length / measuresPerLine);

  console.log(`📈 계산된 줄 수: ${systemCount}, 생성된 줄 이미지 수: ${lineImages.length}`);

  // XML 데이터 (Flutter에서 받아온 원본)
  const xmlBase64 = btoa(unescape(encodeURIComponent(xmlText)));


  window.flutter_inappwebview.callHandler("getDataFromOSMD", 
    sheetImage,
    {
      cursorList,
      bpm: BPM,
      canvasWidth: fullCanvas.width,
      canvasHeight: fullCanvas.height,
      xmlData: xmlBase64,
      lineImages: lineImages,
      lineCount: systemCount,
    }
  );
  isRendered = true;
};