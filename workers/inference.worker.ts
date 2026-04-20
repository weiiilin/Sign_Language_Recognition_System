// @ts-ignore
import * as Comlink from 'comlink'

// @ts-ignore
import * as ort from 'onnxruntime-web/webgpu'
let session: ort.InferenceSession | null = null
let labels: string[] = ['你好', '謝謝', '對不起', '再見', '是', '否', '幫助', '愛', '吃', '喝']

const AIWorker = {
  async loadModel(modelUrl: string) {
    try {
      // 使用 Cache API 實作模型快取
      const cache = await caches.open('onnx-models')
      let response = await cache.match(modelUrl)

      if (!response) {
        response = await fetch(modelUrl)
        await cache.put(modelUrl, response.clone())
      }

      const modelBuffer = await response.arrayBuffer()
      const labelsUrl = new URL('/labels.json', self.location.origin).toString()
      const labelsResponse = await fetch(labelsUrl)
      if (labelsResponse.ok) {
        const labelsPayload = await labelsResponse.json()
        if (Array.isArray(labelsPayload)) {
          labels = labelsPayload.map((label) => String(label))
        } else if (Array.isArray(labelsPayload.classes)) {
          if (Array.isArray(labelsPayload.display_names) && labelsPayload.display_names.length === labelsPayload.classes.length) {
            labels = labelsPayload.display_names.map((label: unknown) => String(label))
          } else {
            labels = labelsPayload.classes.map((label: unknown) => String(label))
          }
        }
      }
      
      // 初始化 ONNX Runtime Session，指定 WebGPU 執行環境
      ort.env.wasm.numThreads = 1; 
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgpu', 'wasm'], 
      })
      
      return true
    } catch (error) {
      console.error('模型載入失敗:', error)
      return false
    }
  },

  async predict(landmarks: number[]) {
    if (!session) return '模型未載入'
    
    try {
      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      if (!inputName || !outputName) return '模型輸入輸出名稱異常'

      // 將 MediaPipe 傳來的節點轉換為 ONNX 需要的 Tensor (假設模型輸入為 1x63 的 Float32Array)
      const inputTensor = new ort.Tensor('float32', new Float32Array(landmarks), [1, landmarks.length])
      const feeds: Record<string, ort.Tensor> = {}
      feeds[inputName] = inputTensor

      const results = await session.run(feeds)
      const outputTensor = results[outputName]
      if (!outputTensor) return '模型輸出異常'
      const output = outputTensor.data as Float32Array
      if (output.length === 0) return '模型輸出為空'
      
      // 簡單的 ArgMax 找出機率最高的類別 (需對應你訓練的 10 個手語標籤)
      let maxIndex = 0
      for (let i = 1; i < output.length; i++) {
        const currentScore = output[i] ?? Number.NEGATIVE_INFINITY
        const maxScore = output[maxIndex] ?? Number.NEGATIVE_INFINITY
        if (currentScore > maxScore) maxIndex = i
      }
      
      return labels[maxIndex] ?? '未知類別'
    } catch (e) {
      return '辨識錯誤'
    }
  }
}

Comlink.expose(AIWorker)
export type AIWorkerType = typeof AIWorker