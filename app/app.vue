<template>
  <div class="container">
    <h1>手語辨識系統</h1>
    <button class="start-btn" :disabled="isStarting" @click="startSystem">
      {{ isStarting ? '系統啟動中...' : '重新啟動系統' }}
    </button>

    <div class="video-wrapper">
      <video ref="videoRef" autoplay playsinline muted></video>
      <canvas ref="canvasRef"></canvas>
    </div>

    <div class="status-panel">
      <h2 v-if="signStore.isModelLoaded">辨識結果: <span class="result">{{ signStore.currentSign }}</span></h2>
      <div v-if="inferenceLogs.length > 0" class="log-container">
        <h3>實時分析 (Confidence)</h3>
        <div v-for="(item, idx) in inferenceLogs" :key="idx" class="log-item">
          <span class="log-label">{{ item.label }}:</span>
          <div class="log-bar-bg">
            <div class="log-bar-fill" :style="{ width: (item.score * 100) + '%' }"></div>
          </div>
          <span class="log-percent">{{ (item.score * 100).toFixed(1) }}%</span>
        </div>
      </div>
      <p class="status-text">{{ systemStatus }}</p>
      <div v-if="!signStore.isModelLoaded" class="loader">正在下載模型 (約 5-20MB)...</div>
      <p v-if="signStore.errorMsg" class="error">{{ signStore.errorMsg }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useSignStore } from '~/../stores/signStore'
import * as Comlink from 'comlink'
import type { AIWorkerType } from '~/../workers/inference.worker'
const framesBuffer = ref<number[][]>([])
const isCollecting = ref(false)

function mirrorX(coords: number[]): number[] {
  const out = coords.slice();
  for (let i = 0; i < out.length; i += 3) {
    out[i] = 1 - out[i]!;
  }
  return out;
}

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const signStore = useSignStore();
const inferenceLogs = ref<{ label: string, score: number }[]>([]);

let workerProxy: Comlink.Remote<AIWorkerType> | null = null
let workerInstance: Worker | null = null
let handLandmarker: any = null
let animationFrameId: number | null = null

const systemStatus = ref('等待啟動...')
const isStarting = ref(false)
let isPredicting = false

// --- 1. 相機權限 ---
const requestCameraAccess = async () => {
  try {
    systemStatus.value = '請求鏡頭權限中...'
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false
    })
    if (videoRef.value) {
      videoRef.value.srcObject = stream
      await new Promise((r) => videoRef.value!.onloadedmetadata = r)
    }
  } catch (err) {
    throw new Error('無法存取相機，請檢查權限設定。')
  }
}

// --- 2. 初始化 Worker & MediaPipe ---
const initSystem = async () => {
  try {
    systemStatus.value = '載入推論引擎中...'
    if (!workerInstance) {
      workerInstance = new Worker(
        new URL('~/../workers/inference.worker.ts', import.meta.url),
        { type: 'module' }
      )
      workerProxy = Comlink.wrap<AIWorkerType>(workerInstance)
    }

    const success = await workerProxy!.loadModel('/model.onnx')
    if (!success) throw new Error('模型載入失敗')
    signStore.setModelLoaded(true)

    systemStatus.value = '載入手部偵測模型...'
    const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision')
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    )
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    })

    systemStatus.value = '系統就緒，請開始比手語！'
    return true
  } catch (error: any) {
    console.error(error)
    signStore.setError(error.message)
    return false
  }
}

// --- 3. 核心偵測與辨識邏輯 ---
const detectFrame = () => {
  if (!videoRef.value || !canvasRef.value || !handLandmarker) return
  const ctx = canvasRef.value.getContext('2d')

  const renderLoop = async () => {
    if (!videoRef.value || !canvasRef.value) return

    // 同步畫布尺寸
    if (canvasRef.value.width !== videoRef.value.videoWidth) {
      canvasRef.value.width = videoRef.value.videoWidth
      canvasRef.value.height = videoRef.value.videoHeight
    }

    const startTimeMs = performance.now()
    const results = handLandmarker.detectForVideo(videoRef.value, startTimeMs)

    ctx?.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)

    // 準備這一影格的資料 (固定為 63 + 63 = 126 點)
    const leftHand = new Array(63).fill(0)
    const rightHand = new Array(63).fill(0)

    if (results.landmarks && results.landmarks.length > 0) {
      for (let i = 0; i < results.landmarks.length; i++) {
        const handInfo = results.handedness[i]?.[0]
        let label = handInfo?.categoryName || handInfo?.label // 取得 Left 或 Right
        console.log(`偵測到 ${label} 手`)
        const handCount = results.landmarks.length;

        // 1. 取得原始座標並進行 X 軸翻轉
        // 我們將每個點的 x 座標用 (1.0 - x) 翻轉，這樣模型看到的左右就跟畫面上看到的一致
        let coords = results.landmarks[i].flatMap((lm: any) => [
         lm.x, // 這裡進行翻轉
          lm.y,
          lm.z
        ]);
        
     if (label === 'Right') {
         coords = mirrorX(coords);
      }

  coords.forEach((val:number, idx:number) => leftHand[idx] = val);
        // // 對調完之後，再放入對應的陣列
        // if (label === 'Left') {
        //   coords.forEach((val: number, idx: number) => leftHand[idx] = val)
        // } else {
        //   coords.forEach((val: number, idx: number) => rightHand[idx] = val);
        //   // coords.forEach((val:number, idx:number) => leftHand[idx] = val);
        // }
      }
    }

    // 將這一幀的 126 個點合併

    const currentFrameData = [...leftHand, ...rightHand]

    if (framesBuffer.value.length >= 30) {
      const frames = [...framesBuffer.value]
      framesBuffer.value = []  // 先清空，不阻塞下一輪收集

      if (workerProxy) {
        // @ts-ignore
        workerProxy.predict(frames).then((res: any) => {
          if (res && typeof res === 'object') {
            inferenceLogs.value = res.allProbabilities.slice(0, 3)
            signStore.updateSign(res.prediction)
            systemStatus.value = `偵測到：${res.prediction} (${(res.confidence * 100).toFixed(0)}%)`
          }
        })
      }
    }
    if (!isPredicting && workerProxy) {
      isPredicting = true

      workerProxy.predict(currentFrameData).then((res: any) => {
        if (res && typeof res === 'object') {
          // 更新 UI 上的機率條 (取前 3 名即可) 
          inferenceLogs.value = res.allProbabilities.slice(0, 3);

          // 更新原本的 store 邏輯
          if (res.prediction !== '辨識中...') {
            signStore.updateSign(res.prediction);
            systemStatus.value = `偵測到：${res.prediction} (${(res.confidence * 100).toFixed(0)}%)`;
          } else {
            systemStatus.value = '動作分析中...';
          }
        }
        isPredicting = false;
      });
    }

    // 繪製骨架點 (方便觀察)
    if (results.landmarks) {
      ctx!.fillStyle = '#00FF00'
      for (const handLandmarks of results.landmarks) {
        for (const landmark of handLandmarks) {
          ctx!.beginPath()
          ctx!.arc(landmark.x * canvasRef.value.width, landmark.y * canvasRef.value.height, 3, 0, 2 * Math.PI)
          ctx!.fill()
        }
      }
    }

    animationFrameId = requestAnimationFrame(renderLoop)
  }

  renderLoop()
}

// --- 4. 啟動與生命週期 ---
const startSystem = async () => {
  if (isStarting.value) return
  isStarting.value = true

  try {
    await requestCameraAccess()
    const ready = await initSystem()
    if (ready) {
      detectFrame()
    }
  } catch (e: any) {
    signStore.setError(e.message)
  } finally {
    isStarting.value = false
  }
}

onMounted(() => {
  startSystem()
})

onUnmounted(() => {
  if (animationFrameId) cancelAnimationFrame(animationFrameId)
  workerInstance?.terminate()
  if (videoRef.value?.srcObject) {
    (videoRef.value.srcObject as MediaStream).getTracks().forEach(t => t.stop())
  }
})
</script>

<style scoped>
.log-container {
  background: rgba(0, 0, 0, 0.05);
  padding: 10px;
  border-radius: 8px;
  margin: 10px auto;
  max-width: 300px;
  text-align: left;
}

.log-item {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
  font-size: 0.9em;
}

.log-label {
  width: 60px;
  font-weight: bold;
}

.log-bar-bg {
  flex: 1;
  background: #ddd;
  height: 10px;
  margin: 0 10px;
  border-radius: 5px;
  overflow: hidden;
}

.log-bar-fill {
  background: #2563eb;
  height: 100%;
  transition: width 0.1s ease-out;
}

.log-percent {
  width: 45px;
  text-align: right;
  font-family: monospace;
}

.container {
  text-align: center;
  padding: 20px;
}

.video-wrapper {
  position: relative;
  display: inline-block;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
}

video {
  width: 640px;
  height: 480px;
  transform: scaleX(-1);
  display: block;
}

canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 640px;
  height: 480px;
  transform: scaleX(-1);
}

.status-panel {
  margin-top: 20px;
  min-height: 100px;
}

.result {
  color: #2563eb;
  font-size: 1.5em;
  text-decoration: underline;
}

.error {
  color: #ef4444;
  font-weight: bold;
}

.start-btn {
  padding: 12px 24px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 20px;
}

.start-btn:disabled {
  background: #94a3b8;
}
</style>