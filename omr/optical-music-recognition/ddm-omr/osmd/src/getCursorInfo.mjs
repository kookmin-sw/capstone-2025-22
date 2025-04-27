import FS from "fs";

const defaultCursorWidth = 30;
const defaultCursorHeight = 40;

const cursorWidth = 20;
const cursorHeight = 40;

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
    top: top + defaultCursorHeight / 2 - cursorHeight / 2,
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
export function extractCursorInfo(cursor) {
  const cursorList = [];
  const measureList = [];

  cursor.CursorOptions.type = 0;
  cursor.resetIterator();
  cursor.show();

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
 * 파일 저장
 */
export function saveJsonData(result, fileName) {
  const jsonData = {
    origin: fileName,
    cursorList: [],
    measureList: [],
  };

  for (const [key, value] of Object.entries(result)) {
    let prevY = value[0].top;
    let temp = [];
    for (var item of value) {
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
  const filePath = `${fileName}.json`;
  var jsonString = JSON.stringify(jsonData, null, 2);
  FS.writeFileSync(filePath, jsonString, { encoding: "utf-8" });
}
