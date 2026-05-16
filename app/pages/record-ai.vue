<template>
  <main class="page">
    <header class="top-bar">
      <button class="setting" @click="showSetting = true">
        ☰
      </button>
      <div class="logo">
        Sign Language 
        Recognition System
      </div>
      
    </header>
  <section class="camera">
      <video ref="videoRef" autoplay playsinline muted></video>
      <canvas ref="canvasRef"></canvas>
      
    </section>
    <button class="start-btn" :disabled="isStarting" @click="startSystem">
        {{ isStarting ? '系統啟動中...' : '重新啟動系統' }}
      </button>
    <div class="pill">
      台灣手語 → 中文（繁體）
    </div>

    <BottomNav />
    <section class="card" @click="showDetail = true">
      <p>辨識結果：{{ signStore.currentSign || '尚未辨識' }}</p>
      <span>{{ systemStatus }}</span>
    </section>

     <SettingModal v-if="showSetting" @close="showSetting = false" />
    <DetailSheet
      v-if="showDetail"
      @close="showDetail = false"
      :word="signStore.currentSign || '尚未辨識'"
      breakdown="依辨識結果顯示拆解"
      detail="手型：待補<br />位置：待補<br />動作：待補"
    />
  </main>
</template>

<script setup lang="ts">

const showSetting = ref(false)
const showDetail = ref(false)

import { ref, onMounted, onUnmounted } from 'vue'
import { useSignStore } from '~/../stores/signStore'
import * as Comlink from 'comlink'
import type { AIWorkerType } from '~/../workers/inference.worker'

const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const signStore = useSignStore();

type PredictionResult = {
  prediction: string
  confidence: number
  allProbabilities: Array<{ label: string; score: number }>
}

function mirrorX(coords: number[]): number[] {
  const out = coords.slice()
  for (let i = 0; i < out.length; i += 3) {
    out[i] = 1 - out[i]!
  }
  return out
}

function handlePredictionResult(res: string | PredictionResult) {
  if (typeof res === 'string') {
    systemStatus.value = res
    return
  }

  if (res.prediction !== '辨識中...') {
    signStore.updateSign(res.prediction)
    systemStatus.value = `偵測到：${res.prediction} (${(res.confidence * 100).toFixed(0)}%)`
  } else {
    systemStatus.value = '動作分析中...'
  }
}

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

    systemStatus.value = '系統就緒，請開始比手語！'
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

    // 準備這一影格的資料，Worker 會在背景收集 30 幀。
    const leftHand = new Array(63).fill(0)
    const rightHand = new Array(63).fill(0)

    if (results.landmarks && results.landmarks.length > 0) {
      for (let i = 0; i < results.landmarks.length; i++) {
        const handInfo = results.handedness[i]?.[0]
        const label = handInfo?.categoryName || handInfo?.label
        let coords = results.landmarks[i].flatMap((lm: any) => [lm.x, lm.y, lm.z])

        if (label === 'Right' || label === 'right') {
          coords = mirrorX(coords)
        }

        coords.forEach((val: number, idx: number) => leftHand[idx] = val)
      }
    }

    const currentFrameData = [...leftHand, ...rightHand]

    if (!isPredicting && workerProxy) {
      isPredicting = true

      workerProxy.predict(currentFrameData).then((res) => {
        handlePredictionResult(res)
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

.camera {
  margin-top: 20px;
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

</style>
