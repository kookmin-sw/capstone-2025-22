const LABELLING_MODE = true;

const defaultCursorWidth = 30;
const defaultCursorHeight = 40;

const cursorWidth = 20;
const cursorHeight = LABELLING_MODE ? defaultCursorHeight : 100;

const cursorDisplayOffsetY = LABELLING_MODE ? 0 : 20;

/**
 * 커서에서 정보 뽑아오기: 필요한 정보만 추출(top, left, width, height,
 * @param {*} cursor
 * @returns
 */
function getCurrentCursorInfo(cursor) {
  const cursorElement = cursor.cursorElement;

  const { height, width } = cursorElement;

  const [top, left] = [cursorElement.style.top, cursorElement.style.left].map(
    (x) => {
      return parseFloat(x.slice(0, -2));
    }
  );

  return {
    top,
    left,
    height,
    width,
    timestamp: cursor.iterator.currentTimeStamp.realValue,
  };
}

/**
 * 해당하는 커서의 정보를 추출한다.
 * @param {*} cursor 커서 인스턴스 (osmd.cursor)
 * @returns timestamp, 커서 위치, 넓이 등
 */
function getCustomizedCursorInfo(cursor) {
  const { top, left, height, width, timestamp } = getCurrentCursorInfo(cursor);

  return {
    top:
      top + defaultCursorHeight / 2 - cursorHeight / 2 - cursorDisplayOffsetY,
    left: left + defaultCursorWidth / 2 - cursorWidth / 2,
    height: cursorHeight,
    width: cursorWidth,
    timestamp,
  };
}

/**
 * 커서를 순회하면서 커서 정보, 마디 정보 수집
 * @param {*} cursor 커서 인스턴스 (osmd.cursor)
 */
function extractCursorInfo(cursor) {
  const cursorList = [];
  const measureList = [];

  cursor.show(); // this would show the cursor on the first note

  try {
    while (!cursor.iterator.endReached) {
      cursorList.push(getCustomizedCursorInfo(cursor));
      cursor.next();
    }
  } catch (e) {
    console.log(e);
    console.log("completed!");
  }
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  cursor.CursorOptions.type = 3;
  cursor.resetIterator();
  cursor.show();
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////

  let currentMeasureInfo = {
    ...getCurrentCursorInfo(cursor),
  };

  try {
    while (!cursor.iterator.endReached) {
      const currentInfo = getCurrentCursorInfo(cursor);

      if (
        currentInfo.top !== currentMeasureInfo.top ||
        currentInfo.left !== currentMeasureInfo.left
      ) {
        measureList.push(currentMeasureInfo);
      }
      currentMeasureInfo = {
        ...currentInfo,
      };

      cursor.next();
    }

    measureList.push(currentMeasureInfo);
    // cursor.hide(); // 커서 안보이게
  } catch (e) {
    console.log(e);
  }

  return { cursorList: cursorList, measureList: measureList };
}

/**
 * XML 파일을 읽어서 OSMD 시스템을 활용하여 시각화 및 정보 추출
 * @param {*} fileData file byte data
 * @returns
 */
async function runOSMD(fileData) {
  var osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("osmdCanvas", {
    // set options here
    backend: "canvas",
    resize: true,
    drawFromMeasureNumber: 1,
    drawUpToMeasureNumber: Number.MAX_SAFE_INTEGER,
    drawTitle: false,
    drawPartNames: false,
    drawingParameters: "compact",
    pageBackgroundColor: "#FFFFFF",
  });

  await osmd.load(fileData, "");
  window.osmd = osmd;
  await osmd.render();

  return extractCursorInfo(osmd.cursor);
}

/**
 * flutter에서 받은 파일 다루기
 * @param {*} file xml, musicxml 파일만 accept.
 */
async function handleFileFromFlutter(file) {
  try {
    /// without file reader
    var result = null;
    if (file.name.match(".*.mxl")) {
      // TODO: 확인 필요.
      result = file.bytes;
    } else {
      result = file.bytes;
    }

    return runOSMD(result);
  } catch (error) {
    console.log(error);
  }
}

function downloadSheetImage(fileName) {
  // canvas 요소 가져오기
  var canvas = document.getElementById("osmdCanvasVexFlowBackendCanvas1");

  // 캔버스 크기 변경
  var newWidth = 1024;
  var newHeight = canvas.height / 2;

  var tempCanvas = document.createElement("canvas");
  tempCanvas.width = newWidth;
  tempCanvas.height = newHeight;
  var tempContext = tempCanvas.getContext("2d");
  tempContext.drawImage(
    canvas,
    0,
    0,
    canvas.width,
    canvas.height,
    0,
    0,
    newWidth,
    newHeight
  );

  // canvas를 이미지로 변환
  var imageData = tempCanvas.toDataURL("image/png");

  // 이미지를 다운로드하기 위해 가상의 링크 생성
  var downloadLink = document.createElement("a");
  downloadLink.href = imageData;
  downloadLink.download = `${fileName}.png`;

  // 클릭하여 다운로드 실행
  downloadLink.click();
}

function downloadJsonData(result, fileName) {
  const jsonData = { cursorList: [], measureList: [] };

  for (const [key, value] of Object.entries(result)) {
    prevY = value[0].top;
    temp = [];
    for (var item of value) {
      console.log(item);
      if (item.top != prevY) {
        jsonData[key].push(temp);
        temp = [];
      }
      temp.push(item);
      prevY = item.top;
    }

    jsonData[key].push(temp);
  }

  // JSON 문자열 생성
  var jsonString = JSON.stringify(jsonData, null, 2);

  // Blob 생성
  var blob = new Blob([jsonString], { type: "application/json" });

  // a 태그 생성
  var downloadLink = document.createElement("a");
  downloadLink.href = URL.createObjectURL(blob);
  downloadLink.download = `${fileName}.json`;

  // 클릭하여 다운로드 실행
  downloadLink.click();
}

/**
 * 디버깅을 위해 사용함.
 * @param {*} evt file change event
 */
function handleFileSelect(evt) {
  var file = evt.target.files[0]; // FileList object

  if (!file.name.match(".*.xml") && !file.name.match(".*.musicxml") && false) {
    alert("You selected a non-xml file. Please select only music xml files.");
  }

  var reader = new FileReader();

  reader.onload = async function (e) {
    const data = await runOSMD(e.target.result);
    downloadJsonData(data, file.name);
    downloadSheetImage(file.name);
  };

  if (file.name.match(".*.mxl")) {
    // have to read as binary, otherwise JSZip will throw ("corrupted zip: missing 37 bytes" or similar)
    reader.readAsBinaryString(file);
  } else {
    reader.readAsText(file);
  }
}

if (LABELLING_MODE) {
  document
    .getElementById("upload")
    .addEventListener("change", handleFileSelect, false);
} else {
  window.addEventListener("flutterInAppWebViewPlatformReady", function (event) {
    window.flutter_inappwebview
      .callHandler("sendFileToOSMD")
      .then(function (inputJson) {
        handleFileFromFlutter(inputJson).then((result) =>
          window.flutter_inappwebview.callHandler("getDataFromOSMD", result)
        );
      });
  });
}
