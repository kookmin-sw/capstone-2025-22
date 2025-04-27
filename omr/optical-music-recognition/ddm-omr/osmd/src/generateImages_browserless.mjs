import Blob from "cross-blob";
import FS from "fs";
import jsdom from "jsdom";
//import headless_gl from "gl"; // this is now imported dynamically in a try catch, in case gl install fails, see #1160
import OSMD from "./opensheetmusicdisplay.min.js"; // window needs to be available before we can require OSMD
import { extractCursorInfo, saveJsonData } from "./getCursorInfo.mjs";
/*
  Render each OSMD sample, grab the generated images, and
  dump them into a local directory as PNG or SVG files.

  inspired by Vexflow's generate_png_images and vexflow-tests.js

  This can be used to generate PNGs or SVGs from OSMD without a browser.
  It's also used with the visual regression test system (using PNGs) in
  `tools/visual_regression.sh`
  (see package.json, used with npm run generate:blessed and generate:current, then test:visual).

  Note: this script needs to "fake" quite a few browser elements, like window, document,
  and a Canvas HTMLElement (for PNG) or the DOM (for SVG)   ,
  which otherwise are missing in pure nodejs, causing errors in OSMD.
  For PNG it needs the canvas package installed.
  There are also some hacks needed to set the container size (offsetWidth) correctly.

  Otherwise you'd need to run a headless browser, which is way slower,
  see the semi-obsolete generateDiffImagesPuppeteerLocalhost.js
*/

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

// global variables
//   (without these being global, we'd have to pass many of these values to the generateSampleImage function)
// eslint-disable-next-line prefer-const
let [
  sourceDir,
  imageFormat,
  pageWidth,
  pageHeight,
  filterRegex,
  mode,
  debugSleepTimeString,
  skyBottomLinePreference,
] = process.argv.slice(2, 10);
imageFormat = imageFormat?.toLowerCase();
if (!sourceDir || (imageFormat !== "png" && imageFormat !== "svg")) {
  console.log(
    "usage: " +
      // eslint-disable-next-line max-len
      "node test/Util/generateImages_browserless.mjs sourceDirectory svg|png [width|0] [height|0] [filterRegex|all|allSmall] [--debug|--osmdtesting] [debugSleepTime]"
  );
  console.log(
    "  (use pageWidth and pageHeight 0 to not divide the rendering into pages (endless page))"
  );
  console.log(
    '  (use "all" to skip filterRegex parameter. "allSmall" with --osmdtesting skips two huge OSMD samples that take forever to render)'
  );
  console.log(
    "example: node ./generateImages_browserless.mjs ./osmd-dataset png"
  );
  console.log(
    "Error: need osmdBuildDir, sourceDir, resultDir and svg|png arguments. Exiting."
  );
  process.exit(1);
}
let resultDir = sourceDir;
const useWhiteTabNumberBackground = true;
// use white instead of transparent background for tab numbers for PNG export.
//   can fix black rectangles displayed, depending on your image viewer / program.
//   though this is unnecessary if your image viewer displays transparent as white

let pageFormat;

if (!mode) {
  mode = "";
}

// let OSMD; // can only be required once window was simulated
// eslint-disable-next-line @typescript-eslint/no-var-requires

async function init() {
  debug("init");

  const DEBUG = mode.startsWith("--debug");
  // const debugSleepTime = Number.parseInt(process.env.GENERATE_DEBUG_SLEEP_TIME) || 0; // 5000 works for me [sschmidTU]
  if (DEBUG) {
    // debug(' (note that --debug slows down the script by about 0.3s per file, through logging)')
    const debugSleepTimeMs = Number.parseInt(debugSleepTimeString, 10);
    if (debugSleepTimeMs > 0) {
      debug("debug sleep time: " + debugSleepTimeString);
      await sleep(Number.parseInt(debugSleepTimeMs, 10));
      // [VSCode] apparently this is necessary for the debugger to attach itself in time before the program closes.
      // sometimes this is not enough, so you may have to try multiple times or increase the sleep timer. Unfortunately debugging nodejs isn't easy.
    }
  }
  debug("sourceDir: " + sourceDir, DEBUG);
  debug("resultDir: " + resultDir, DEBUG);
  debug("imageFormat: " + imageFormat, DEBUG);

  pageFormat = "Endless";
  pageWidth = Number.parseInt(pageWidth, 10);
  pageHeight = Number.parseInt(pageHeight, 10);
  const endlessPage = !(pageHeight > 0 && pageWidth > 0);
  if (!endlessPage) {
    pageFormat = `${pageWidth}x${pageHeight}`;
  }

  // ---- hacks to fake Browser elements OSMD and Vexflow need, like window, document, and a canvas HTMLElement ----
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const dom = new jsdom.JSDOM("<!DOCTYPE html></html>");
  // eslint-disable-next-line no-global-assign
  // window = dom.window;
  // eslint-disable-next-line no-global-assign
  // document = dom.window.document;

  // eslint-disable-next-line no-global-assign
  global.window = dom.window;
  // eslint-disable-next-line no-global-assign
  global.document = window.document;
  //window.console = console; // probably does nothing
  global.HTMLElement = window.HTMLElement;
  global.HTMLAnchorElement = window.HTMLAnchorElement;
  global.XMLHttpRequest = window.XMLHttpRequest;
  global.DOMParser = window.DOMParser;
  global.Node = window.Node;
  if (imageFormat === "png") {
    global.Canvas = window.Canvas;
  }

  // For WebGLSkyBottomLineCalculatorBackend: Try to import gl dynamically
  //   this is so that the script doesn't fail if gl could not be installed,
  //   which can happen in some linux setups where gcc-11 is installed, see #1160
  try {
    const { default: headless_gl } = await import("gl");
    const oldCreateElement = document.createElement.bind(document);
    document.createElement = function (tagName, options) {
      if (tagName.toLowerCase() === "canvas") {
        const canvas = oldCreateElement(tagName, options);
        const oldGetContext = canvas.getContext.bind(canvas);
        canvas.getContext = function (contextType, contextAttributes) {
          if (
            contextType.toLowerCase() === "webgl" ||
            contextType.toLowerCase() === "experimental-webgl"
          ) {
            const gl = headless_gl(
              canvas.width,
              canvas.height,
              contextAttributes
            );
            gl.canvas = canvas;
            return gl;
          } else {
            return oldGetContext(contextType, contextAttributes);
          }
        };
        return canvas;
      } else {
        return oldCreateElement(tagName, options);
      }
    };
  } catch {
    if (skyBottomLinePreference === "--webgl") {
      debug(
        "WebGL image generation was requested but gl is not installed; using non-WebGL generation."
      );
    }
  }

  // fix Blob not found (to support external modules like is-blob)
  global.Blob = Blob;

  const div = document.createElement("div");
  div.id = "browserlessDiv";
  document.body.appendChild(div);
  // const canvas = document.createElement('canvas')
  // div.canvas = document.createElement('canvas')

  const zoom = 1.0;
  // width of the div / PNG generated
  let width = pageWidth * zoom;
  // TODO sometimes the width is way too small for the score, may need to adjust zoom.

  // if (endlessPage) {
  //   width = width;
  // }
  let height = pageHeight;
  if (endlessPage) {
    height = 32767;
  }
  div.width = width;
  div.height = height;
  // div.offsetWidth = width; // doesn't work, offsetWidth is always 0 from this. see below
  // div.clientWidth = width;
  // div.clientHeight = height;
  // div.scrollHeight = height;
  // div.scrollWidth = width;
  div.setAttribute("width", width);
  div.setAttribute("height", height);
  div.setAttribute("offsetWidth", width);
  // debug('div.offsetWidth: ' + div.offsetWidth, DEBUG) // 0 here, set correctly later
  // debug('div.height: ' + div.height, DEBUG)

  // hack: set offsetWidth reliably
  Object.defineProperties(window.HTMLElement.prototype, {
    offsetLeft: {
      get: function () {
        return parseFloat(window.getComputedStyle(this).marginTop) || 0;
      },
    },
    offsetTop: {
      get: function () {
        return parseFloat(window.getComputedStyle(this).marginTop) || 0;
      },
    },
    offsetHeight: {
      get: function () {
        return height;
      },
    },
    offsetWidth: {
      get: function () {
        return width;
      },
    },
  });
  debug("div.offsetWidth: " + div.offsetWidth, DEBUG);
  debug("div.height: " + div.height, DEBUG);
  // ---- end browser hacks (hopefully) ----

  // load globally

  // Create the image directory if it doesn't exist.
  FS.mkdirSync(resultDir, { recursive: true });

  const sourceDirList = FS.readdirSync(sourceDir, {
    recursive: true,
    withFileTypes: true,
  });
  let samplesToProcess = [];
  const fileEndingRegex = "^.*(([.]xml)|([.]musicxml)|([.]mxl))$";
  for (const dir of sourceDirList) {
    if (!dir.isDirectory()) {
      continue;
    }
    const fileList = FS.readdirSync(`${sourceDir}/${dir.name}`);
    for (const file of fileList) {
      if (file.match(fileEndingRegex)) {
        // debug('found musicxml/mxl: ' + dir)
        samplesToProcess.push(`${sourceDir}/${dir.name}/${file}`);
      } else {
        debug("discarded dir/directory: " + file, DEBUG);
      }
    }
  }

  // filter samples to process by regex if given
  if (filterRegex && filterRegex !== "" && filterRegex !== "all") {
    debug("filtering samples for regex: " + filterRegex, DEBUG);
    samplesToProcess = samplesToProcess.filter(
      (filename) =>
        filename.match(filterRegex) && filename.match(fileEndingRegex)
    );
    debug(`found ${samplesToProcess.length} matches: `, DEBUG);
    for (let i = 0; i < samplesToProcess.length; i++) {
      debug(samplesToProcess[i], DEBUG);
    }
  }

  const backend = imageFormat === "png" ? "canvas" : "svg";
  const osmdInstance = new OSMD.OpenSheetMusicDisplay(div, {
    backend: backend,
    pageBackgroundColor: "#FFFFFF",
    pageFormat: pageFormat,
    resize: true,
    drawFromMeasureNumber: 1,
    drawUpToMeasureNumber: Number.MAX_SAFE_INTEGER,
    drawTitle: false,
    drawSubtitle: false,
    drawPartNames: false,
    drawComposer: false,
    drawingParameters: "compact",
  });
  // for more options check OSMDOptions.ts

  // you can set finer-grained rendering/engraving settings in EngravingRules:
  // osmdInstance.EngravingRules.TitleTopDistance = 5.0 // 5.0 is default
  //   (unless in osmdTestingMode, these will be reset with drawingParameters default)
  // osmdInstance.EngravingRules.PageTopMargin = 5.0 // 5 is default
  // osmdInstance.EngravingRules.PageBottomMargin = 5.0 // 5 is default. <5 can cut off scores that extend in the last staffline
  // note that for now the png and canvas will still have the height given in the script argument,
  //   so even with a margin of 0 the image will be filled to the full height.
  // osmdInstance.EngravingRules.PageLeftMargin = 5.0 // 5 is default
  // osmdInstance.EngravingRules.PageRightMargin = 5.0 // 5 is default
  // osmdInstance.EngravingRules.MetronomeMarkXShift = -8; // -6 is default
  // osmdInstance.EngravingRules.DistanceBetweenVerticalSystemLines = 0.15; // 0.35 is default
  // for more options check EngravingRules.ts (though not all of these are meant and fully supported to be changed at will)

  if (useWhiteTabNumberBackground && backend === "png") {
    osmdInstance.EngravingRules.pageBackgroundColor = "#FFFFFF";
    // fix for tab number having black background depending on image viewer
    //   otherwise, the rectangle is transparent, which can be displayed as black in certain programs
  }
  if (DEBUG) {
    osmdInstance.setLogLevel("debug");
    // debug(`osmd PageFormat: ${osmdInstance.EngravingRules.PageFormat.width}x${osmdInstance.EngravingRules.PageFormat.height}`)
    debug(
      `osmd PageFormat idString: ${osmdInstance.EngravingRules.PageFormat.idString}`
    );
    debug("PageHeight: " + osmdInstance.EngravingRules.PageHeight);
  } else {
    osmdInstance.setLogLevel("info"); // doesn't seem to work, log.debug still logs
  }

  debug(
    "[OSMD.generateImages] starting loop over samples, saving images to " +
      resultDir,
    DEBUG
  );

  // 총 작업량 설정
  console.log(`found ${samplesToProcess.length} files...`);

  for (let i = 0; i < samplesToProcess.length; i++) {
    const sampleFilePath = samplesToProcess[i];
    drawProgress(i, samplesToProcess.length, sampleFilePath);
    debug("sampleFilePath: " + sampleFilePath, DEBUG);

    await generateSampleImage(sampleFilePath, osmdInstance, DEBUG);
  }
  debug("done, exiting.");
}

// eslint-disable-next-line
// let maxRss = 0, maxRssFilename = '' // to log memory usage (debug)
async function generateSampleImage(
  sampleFilePath,
  osmdInstance,
  DEBUG = false
) {
  const filePathName = sampleFilePath.replace(".xml", "");
  let loadParameter = FS.readFileSync(sampleFilePath);

  if (sampleFilePath.endsWith(".mxl")) {
    loadParameter = await OSMD.MXLHelper.MXLtoXMLstring(loadParameter);
  } else {
    loadParameter = loadParameter.toString();
  }
  // debug('loadParameter: ' + loadParameter)
  // debug('typeof loadParameter: ' + typeof loadParameter)

  // set sample-specific options for OSMD visual regression testing
  let isTestOctaveShiftInvisibleInstrument;
  let isTestInvisibleMeasureNotAffectingLayout;

  try {
    debug("loading sample " + sampleFilePath, DEBUG);
    await osmdInstance.load(loadParameter, sampleFilePath); // if using load.then() without await, memory will not be freed up between renders
    if (isTestOctaveShiftInvisibleInstrument) {
      osmdInstance.Sheet.Instruments[0].Visible = false;
    }
    if (isTestInvisibleMeasureNotAffectingLayout) {
      if (osmdInstance.Sheet.Instruments[1]) {
        // some systems can't handle ?. in this script (just a safety check anyways)
        osmdInstance.Sheet.Instruments[1].Visible = false;
      }
    }
  } catch (ex) {
    debug(
      "couldn't load sample " + sampleFilePath + ", skipping. Error: \n" + ex
    );
    return;
  }
  debug("xml loaded", DEBUG);
  try {
    await osmdInstance.render();
    const result = extractCursorInfo(osmdInstance.cursor);
    saveJsonData(result, filePathName);
    // there were reports that await could help here, but render isn't a synchronous function, and it seems to work. see #932
  } catch (ex) {
    debug("renderError: " + ex);
  }
  debug("rendered", DEBUG);

  const markupStrings = []; // svg
  const dataUrls = []; // png
  let canvasImage;

  for (
    let pageNumber = 1;
    pageNumber < Number.POSITIVE_INFINITY;
    pageNumber++
  ) {
    if (imageFormat === "png") {
      canvasImage = document.getElementById(
        "osmdCanvasVexFlowBackendCanvas" + pageNumber
      );
      if (!canvasImage) {
        break;
      }
      if (!canvasImage.toDataURL) {
        debug(
          `error: could not get canvas image for page ${pageNumber} for file: ${sampleFilePath}`
        );
        break;
      }
      dataUrls.push(canvasImage.toDataURL());
    } else if (imageFormat === "svg") {
      const svgElement = document.getElementById("osmdSvgPage" + pageNumber);
      if (!svgElement) {
        break;
      }
      // The important xmlns attribute is not serialized unless we set it here
      svgElement.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      markupStrings.push(svgElement.outerHTML);
    }
  }

  for (
    let pageIndex = 0;
    pageIndex < Math.max(dataUrls.length, markupStrings.length);
    pageIndex++
  ) {
    const pageNumberingString = `${pageIndex + 1}`;
    // pageNumberingString = dataUrls.length > 0 ? pageNumberingString : '' // don't put '_1' at the end if only one page. though that may cause more work
    const pageFilename = `${filePathName}_${pageNumberingString}.${imageFormat}`;

    if (imageFormat === "png") {
      const dataUrl = dataUrls[pageIndex];
      if (!dataUrl || !dataUrl.split) {
        debug(
          `error: could not get dataUrl (imageData) for page ${
            pageIndex + 1
          } of sample: ${pageFilename}`
        );
        continue;
      }
      const imageData = dataUrl.split(";base64,").pop();
      const imageBuffer = Buffer.from(imageData, "base64");

      debug("got image data, saving to: " + pageFilename, DEBUG);
      FS.writeFileSync(pageFilename, imageBuffer, { encoding: "base64" });
    } else if (imageFormat === "svg") {
      const markup = markupStrings[pageIndex];
      if (!markup) {
        debug(
          `error: could not get markup (SVG data) for page ${
            pageIndex + 1
          } of sample: ${pageFilename}`
        );
        continue;
      }

      debug("got svg markup data, saving to: " + pageFilename, DEBUG);
      FS.writeFileSync(pageFilename, markup, { encoding: "utf-8" });
    }

    // debug: log memory usage
    // const usage = process.memoryUsage()
    // for (const entry of Object.entries(usage)) {
    //     if (entry[0] === 'rss') {
    //         if (entry[1] > maxRss) {
    //             maxRss = entry[1]
    //             maxRssFilename = pageFilename
    //         }
    //     }
    //     debug(entry[0] + ': ' + entry[1] / (1024 * 1024) + 'mb')
    // }
    // debug('maxRss: ' + (maxRss / 1024 / 1024) + 'mb' + ' for ' + maxRssFilename)
  }
  // debug('maxRss total: ' + (maxRss / 1024 / 1024) + 'mb' + ' for ' + maxRssFilename)

  // await sleep(5000)
  // }) // end read file
}

function debug(msg, debugEnabled = true) {
  if (debugEnabled) {
    console.log("[generateImages] " + msg);
  }
}

function drawProgress(value, total, currentFileName = "") {
  const barCompleteChar = "\u2588";
  const barIncompleteChar = "\u2591";

  const percentage = (value / total) * 100;
  const percentageInt = Number(percentage);
  const bar =
    barCompleteChar.repeat(percentageInt) +
    barIncompleteChar.repeat(100 - percentageInt);

  console.log(
    `Progress |${bar}| ${percentage}% || ${value}/${total} || ${currentFileName}`
  );
}

init();
