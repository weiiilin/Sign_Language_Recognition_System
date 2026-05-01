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

// --- 1. 全域變數與緩衝區設定 ---
const SEQUENCE_LENGTH = 30; // 模型要 30 幀 (30 * 126 = 3780)
let frameBuffer: number[][] = []; // 用來存最近 30 幀的資料

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const signStore = useSignStore();

let workerProxy: Comlink.Remote<AIWorkerType> | null = null
let workerInstance: Worker | null = null
let handLandmarker: any = null
let animationFrameId: number | null = null

const systemStatus = ref('等待啟動...')
const isStarting = ref(false)
let isPredicting = false

// --- 2. 相機權限 ---
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

// --- 3. 初始化 Worker & MediaPipe ---
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

    systemStatus.value = '系統就緒'
    return true
  } catch (error: any) {
    console.error(error)
    signStore.setError(error.message)
    return false
  }
}

// --- 4. 核心偵測與辨識邏輯 ---
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

    // 準備這一影格的資料
    const leftHand = new Array(63).fill(0)
    const rightHand = new Array(63).fill(0)

    if (results.landmarks && results.landmarks.length > 0) {
      for (let i = 0; i < results.landmarks.length; i++) {
        const handInfo = results.handedness[i]?.[0]
        const label = handInfo?.categoryName || handInfo?.label
        const coords = results.landmarks[i].flatMap((lm: any) => [lm.x, lm.y, lm.z])

        if (label === 'Left' || label === 'left') {
          coords.forEach((val: number, idx: number) => leftHand[idx] = val)
        } else {
          coords.forEach((val: number, idx: number) => rightHand[idx] = val)
        }
      }
    }

    // 更新緩衝區
    const currentFrameData = [...leftHand, ...rightHand]
    frameBuffer.push(currentFrameData)

    // 保持緩衝區只有 30 幀
    if (frameBuffer.length > SEQUENCE_LENGTH) {
      frameBuffer.shift()
    }

    // 湊滿 30 幀就送去推論 (30 * 126 = 3780)
    if (frameBuffer.length === SEQUENCE_LENGTH && !isPredicting && workerProxy) {
      isPredicting = true
      const flattenedData = frameBuffer.flat() // 攤平成 3780 個數字

      workerProxy.predict(flattenedData).then((res) => {
        signStore.updateSign(res)
        isPredicting = false
      }).catch((err) => {
        console.error("預測失敗:", err)
        isPredicting = false
      })
    }

    // 繪製骨架點 (方便觀察是否有偵測到手)
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

// --- 5. 啟動與生命週期 ---
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