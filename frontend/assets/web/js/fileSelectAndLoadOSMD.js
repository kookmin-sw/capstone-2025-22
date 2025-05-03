// 커서 스타일
const cursorWidth = 20;
const cursorHeight = 100;

// 전역 변수 정의
let xmlData = "";
let BPM = 60; // 기본 BPM 값
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
  drawingParameters: "default",
  drawMeasureNumbers: true,
  pageBackgroundColor: "transparent",
  renderSingleHorizontalStaffline: false,
};

// 악보 전체 이미지 만드는 함수
async function createSheetImage(canvas, maxWidth) {
  const scale = Math.min(1.0, maxWidth / canvas.width);
  const newWidth = canvas.width * scale;
  const newHeight = canvas.height * scale;
  const dpr = window.devicePixelRatio;

  const tempCanvas = document.createElement("canvas");
  const ctx = tempCanvas.getContext("2d");

  tempCanvas.width = newWidth * dpr;
  tempCanvas.height = newHeight * dpr;
  tempCanvas.style.width = `${newWidth}px`;
  tempCanvas.style.height = `${newHeight}px`;

  ctx.scale(dpr, dpr);
  ctx.drawImage(
    canvas,
    0, 0, canvas.width, canvas.height, // source
    0, 0, newWidth, newHeight // destination
  );

  return tempCanvas.toDataURL("image/png").split(",")[1];
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
  const canvas = document.getElementById("osmdCanvasVexFlowBackendCanvas1");
  const canvasRect = canvas.getBoundingClientRect();
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

  osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem = 4; // 한 줄에 4마디
  osmd.EngravingRules.PageTopMargin = 0;
  osmd.EngravingRules.PageBottomMargin = 0;

  await osmd.load(xmlText, "");
  window.osmd = osmd;
  await osmd.render();

  osmd.cursor.reset()
  osmd.cursor.hide();
  await new Promise(requestAnimationFrame);

  // 렌더링 완료 후 SheetImage, 데이터 수집
  const canvas = document.getElementById("osmdCanvasVexFlowBackendCanvas1");
  const sheetImage = await createSheetImage(canvas, 1080);

  // 커서 리스트 수집
  const cursorList = getCursorList(osmd.cursor);

  // 악보 줄 수: MusicXML의 <measure> 태그 개수를 세어서 4마디씩 나눈 줄 수로 계산
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlText, 'application/xml');
  const measures = xmlDoc.getElementsByTagName('measure');
  const measuresPerLine = osmd.EngravingRules.RenderXMeasuresPerLineAkaSystem || 4;
  const systemCount = Math.ceil(measures.length / measuresPerLine);

  // XML 데이터 (Flutter에서 받아온 원본)
  const xmlBase64 = btoa(unescape(encodeURIComponent(xmlText)));

  console.log("📌 [JS] cursorList length:", cursorList.length);
  console.log("📌 [JS] sample cursor:", cursorList[0]);
  console.log("📌 [JS] systemCount:", systemCount);

  window.flutter_inappwebview.callHandler("getDataFromOSMD", 
    sheetImage,
    {
      cursorList,
      bpm: BPM,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      xmlData: xmlBase64,
      lineCount: systemCount,
    }
  );

  isRendered = true;
  
};

