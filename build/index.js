import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, drawSkeleton, isMobile, drawHeadVectors, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';
import { modelFromJSON } from '@tensorflow/tfjs-layers/dist/models';
import "babel-polyfill"
const videoWidth = window.innerWidth; //600;
const videoHeight = window.innerHeight;//500;
const stats = new Stats();
/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }
  
    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;
  
    const mobile = isMobile();
    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        width: mobile ? undefined : videoWidth,
        height: mobile ? undefined : videoHeight,
      },
    });
    video.srcObject = stream;
  
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  }
  async function loadVideo() {
    const video = await setupCamera();
    video.play();
  
    return video;
  }

  // some default values that we need
  // to conjure up the model
const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 257;
const isModelLoaded = false;
const wasModelLoaded = false;
const modelState = {
    algorithm: 'multi-pose',
    input: {
      architecture: 'MobileNetV1',
      outputStride: defaultMobileNetStride,
      inputResolution: defaultMobileNetInputResolution,
      multiplier: defaultMobileNetMultiplier,
      quantBytes: defaultQuantBytes
    },
    singlePoseDetection: {
      minPoseConfidence: 0.1,
      minPartConfidence: 0.5,
    },
    multiPoseDetection: {
      maxPoseDetections: 5,
      minPoseConfidence: 0.15,
      minPartConfidence: 0.1,
      nmsRadius: 30.0,
    },
    output: {
      showVideo: true,
      showSkeleton: true,
      showPoints: true,
      showBoundingBox: false,
    },
    net: null,
  };
/**
 * Sets up a frames per second panel on the top-left of the window
*/
function setupFPS() {
    stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
    document.getElementById('main').appendChild(stats.dom);
}
/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
*/
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  const flipPoseHorizontal = true;
  
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  async function poseDetectionFrame() {
    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;

    let all_poses = await net.estimatePoses(video,{
      flipHorizontal: flipPoseHorizontal,
      decodingMethod: 'multi-person',
      maxDetections: modelState.multiPoseDetection.maxPoseDetections,
      scoreThreshold: modelState.multiPoseDetection.minPartConfidence,
      nmsRadius: modelState.multiPoseDetection.nmsRadius
    });
    poses = poses.concat(all_poses);
    minPoseConfidence = +modelState.multiPoseDetection.minPoseConfidence;
    minPartConfidence = +modelState.multiPoseDetection.minPartConfidence;
  
  ctx.clearRect(0, 0, videoWidth, videoHeight);
    //console.log(poses)
    if (modelState.output.showVideo) {
      
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }
    // End monitoring code for frames per second
    stats.end();

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        if (modelState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (modelState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
          drawHeadVectors(keypoints, minPartConfidence, ctx);
        }
        if (modelState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    requestAnimationFrame(poseDetectionFrame);
  }
  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
    //load the model
    const net = await posenet.load({
        architecture: modelState.input.architecture,
        outputStride: modelState.input.outputStride,
        inputResolution: modelState.input.inputResolution,
        multiplier:modelState.input.multiplier,
        quantBytes:modelState.input.quantBytes
    });
    let video;
    try {
        video = await loadVideo();
    } catch (e) {
      let info = document.getElementById('info');
      info.textContent = 'this browser does not support video capture,' +
          'or this device does not have a camera';
      info.style.display = 'block';
      throw e;
}
let info = document.getElementById('loading');
info.textContent = "Model Loaded"
setupFPS();
detectPoseInRealTime(video, net);
}
navigator.getUserMedia = navigator.getUserMedia ||
navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();