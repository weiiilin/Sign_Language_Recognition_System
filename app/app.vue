<template>
  <div class="container">
    <h1>手語辨識系統</h1>
    <button class="start-btn" :disabled="isStarting" @click="startSystem">
      {{ isStarting ? '啟動中...' : '啟用鏡頭並開始辨識' }}
    </button>
    <div class="video-wrapper">
      <video ref="videoRef" autoplay playsinline muted></video>
      <canvas ref="canvasRef"></canvas>
    </div>

    <div class="status-panel">
      <h2>目前手語: {{ signStore.currentSign }}</h2>
      <p>{{ systemStatus }}</p>
      <p v-if="!signStore.isModelLoaded">模型載入中，請稍候...</p>
      <p v-if="signStore.errorMsg" class="error">{{ signStore.errorMsg }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useSignStore } from '~/../stores/signStore'
import * as Comlink from 'comlink'
import type { AIWorkerType } from '~/../workers/inference.worker'

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const signStore = useSignStore()

let worker: Comlink.Remote<AIWorkerType> | null = null
let workerInstance: Worker | null = null
let handLandmarker: any = null
let animationFrameId: number | null = null
const systemStatus = ref('系統準備中...')
const isStarting = ref(false)

const requestCameraAccess = async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('此瀏覽器不支援相機存取，請改用最新版本的 Chrome 或 Edge。')
  }

  if (!window.isSecureContext) {
    throw new Error('目前不是安全來源，請改用 https 或 localhost 存取，才能使用鏡頭。')
  }

  systemStatus.value = '請求鏡頭權限中...'
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
    audio: false
  })

  if (!videoRef.value) {
    stream.getTracks().forEach(track => track.stop())
    throw new Error('找不到影片元件，無法啟用鏡頭。')
  }

  videoRef.value.srcObject = stream
  await new Promise<void>((resolve) => {
    videoRef.value!.onloadedmetadata = () => resolve()
  })
  await videoRef.value.play()
  systemStatus.value = '鏡頭已啟用，開始辨識...'
}

const initSystem = async () => {
  try {
    if (!import.meta.client) return false

    // 1. 初始化 Web Worker 與 ONNX 模型
    systemStatus.value = '初始化推論引擎中...'
    workerInstance = new Worker(new URL('../workers/inference.worker.ts', import.meta.url), { type: 'module' })
    worker = Comlink.wrap<AIWorkerType>(workerInstance)
    
    // 假設你訓練好的模型放在 public/model.onnx
    const modelLoaded = await worker.loadModel('/model.onnx')
    if (!modelLoaded) {
      throw new Error('模型載入失敗，請檢查 /public/model.onnx')
    }
    signStore.setModelLoaded(modelLoaded)

    // 2. 初始化 MediaPipe Hand Landmarker
    systemStatus.value = '初始化手部追蹤中...'
    const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision')
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    )
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU" // 使用 WebGL 加速 MediaPipe
      },
      runningMode: "VIDEO",
      numHands: 2
    })
  } catch (error: any) {
    signStore.setError(`系統初始化失敗: ${error?.message || '未知錯誤'}`)
    return false
  }

  return true
}

const startSystem = async () => {
  if (!import.meta.client) return
  if (isStarting.value) return

  signStore.setError('')
  isStarting.value = true

  try {
    await requestCameraAccess()

    const initialized = await initSystem()
    if (!initialized) return

    detectFrame()
  } catch (error: any) {
    signStore.setError(`鏡頭啟動失敗: ${error?.message || '請確認瀏覽器權限設定。'}`)
    systemStatus.value = '初始化失敗'
  } finally {
    isStarting.value = false
  }
}

const detectFrame = async () => {
  if (!videoRef.value || !canvasRef.value || !handLandmarker) return
  
  const video = videoRef.value
  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')
  
  // 確保 Canvas 尺寸與 Video 一致
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  let lastVideoTime = -1

  const renderLoop = async () => {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime
      
      // MediaPipe 偵測手部節點
      const results = handLandmarker!.detectForVideo(video, performance.now())
      
      // 繪製影像到 Canvas
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      ctx?.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      if (results.landmarks && results.landmarks.length > 0) {
        // 提取第一個手的所有節點 (21個點 * 3維度 = 63個數值)
        const flatLandmarks = results.landmarks[0]?.flatMap((p: any) => [p.x, p.y, p.z])
        
        // 透過 Comlink 將數據非同步送給 Web Worker 進行 ONNX 推論
        if (worker && signStore.isModelLoaded && flatLandmarks) {
          const prediction = await worker.predict(flatLandmarks)
          if (prediction) {
            signStore.updateSign(prediction)
          }
        }

        // 這裡可以加入 Web Canvas API 繪製骨架的程式碼
        // drawConnectors(ctx, results.landmarks[0], HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 5})
      }
    }
    animationFrameId = requestAnimationFrame(renderLoop)
  }
  systemStatus.value = '辨識中...'
  renderLoop()
}

onMounted(() => {
  startSystem()
})

onUnmounted(() => {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId)
  }

  if (videoRef.value && videoRef.value.srcObject) {
    const tracks = (videoRef.value.srcObject as MediaStream).getTracks()
    tracks.forEach(track => track.stop())
  }

  workerInstance?.terminate()
})
</script>

<style scoped>
.container { text-align: center; font-family: sans-serif; }
.start-btn {
  margin-bottom: 12px;
  padding: 10px 16px;
  border: none;
  border-radius: 8px;
  background: #2563eb;
  color: #fff;
  cursor: pointer;
  font-weight: 600;
}
.start-btn:disabled { opacity: 0.7; cursor: not-allowed; }
.video-wrapper { position: relative; display: inline-block; }
video {
  width: 640px;
  height: 480px;
  border: 2px solid #333;
  border-radius: 8px;
  transform: scaleX(-1);
  display: block;
}
canvas {
  position: absolute;
  top: 0;
  left: 0;
  border: 2px solid #333;
  border-radius: 8px;
  transform: scaleX(-1);
  pointer-events: none;
}
.status-panel { margin-top: 20px; }
.error { color: red; }
</style>