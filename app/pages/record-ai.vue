<template>
  <main class="page">

    <section class="camera">
      <button class="setting">☰</button>

      <video ref="videoRef" autoplay playsinline muted></video>
      <canvas ref="canvasRef"></canvas>

      <button class="start-btn" :disabled="isStarting" @click="startSystem">
        {{ isStarting ? '系統啟動中...' : '重新啟動系統' }}
      </button>
    </section>

    <div class="pill">
      台灣手語 → 中文（繁體）
    </div>

    <BottomNav />

    <section class="card" @click="showDetail = true">
      <p>辨識結果：{{ signStore.currentSign || '尚未辨識' }}</p>
      <span>{{ systemStatus }}</span>
    </section>

     <SettingModal v-if="showSetting" @close="showSetting = false" />
    <DetailSheet v-if="showDetail" @close="showDetail = false" />


  </main>
</template>

<script setup lang="ts">

import { ref, onMounted, onUnmounted } from 'vue'
import { useSignStore } from '~/../stores/signStore' // 請確認此路徑正確
import * as Comlink from 'comlink'
import type { AIWorkerType } from '~/../workers/inference.worker'

const showSetting = ref(false)
const showDetail = ref(false)

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

const initSystem = async () => {
  try {
    // 1. 初始化 Worker
    systemStatus.value = '載入推論引擎中...'
    if (!workerInstance) {
      // 使用 Vite 的 import.meta.url 語法載入 Worker
      workerInstance = new Worker(
        new URL('~/../workers/inference.worker.ts', import.meta.url),
        { type: 'module' }
      )
      workerProxy = Comlink.wrap<AIWorkerType>(workerInstance)
    }

    // 2. 載入模型 (請確認 model.onnx 放在 public 根目錄下)
    const success = await workerProxy!.loadModel('/model.onnx')
    if (!success) throw new Error('模型載入失敗')
    signStore.setModelLoaded(true)

    // 3. 初始化 MediaPipe
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
      numHands: 1
    })

    systemStatus.value = '系統就緒'
    return true
  } catch (error: any) {
    console.error(error)
    signStore.setError(error.message)
    return false
  }
}

const detectFrame = () => {
  if (!videoRef.value || !canvasRef.value || !handLandmarker) return

  const ctx = canvasRef.value.getContext('2d')

  const renderLoop = async () => {
    if (!videoRef.value || !canvasRef.value) return

    // 確保尺寸一致
    if (canvasRef.value.width !== videoRef.value.videoWidth) {
      canvasRef.value.width = videoRef.value.videoWidth
      canvasRef.value.height = videoRef.value.videoHeight
    }

    const startTimeMs = performance.now()
    const results = handLandmarker.detectForVideo(videoRef.value, startTimeMs)

    ctx?.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)

    if (results.landmarks && results.landmarks.length > 0) {
      // 提取座標並推論 (限制推論頻率以免阻塞)
      if (!isPredicting && workerProxy && signStore.isModelLoaded) {
        const landmarks = results.landmarks[0].flatMap((p: any) => [p.x, p.y, p.z])

        isPredicting = true
        workerProxy.predict(landmarks).then((res) => {
          signStore.updateSign(res)
          isPredicting = false
        }).catch(() => isPredicting = false)
      }

      // 繪製簡單的提示點
      ctx!.fillStyle = '#00FF00'
      for (const landmark of results.landmarks[0]) {
        ctx!.beginPath()
        ctx!.arc(landmark.x * canvasRef.value.width, landmark.y * canvasRef.value.height, 3, 0, 2 * Math.PI)
        ctx!.fill()
      }
    }

    animationFrameId = requestAnimationFrame(renderLoop)
  }

  renderLoop()
}
// 啟動系統的流程：請求相機權限 -> 初始化 Worker 和模型 -> 啟動偵測循環
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
.page {
  min-height: 100vh;
  background: white;
  padding: 20px;
}

.camera {
  height: 400px;
  background: #eee;
  border-radius: 24px;
  position: relative;
  overflow: hidden;
}

video,
canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

.setting {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 3;
}

.start-btn {
  margin: 12px auto;
  position: absolute;
  background: #1e8cff;
  color: white;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
}

.pill {
  margin: 12px auto;
  background: #d8ecff;
  color: #2488e8;
  padding: 6px 12px;
  border-radius: 999px;
  width: fit-content;
}

.card {
  margin-top: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 16px;
}
</style>