import * as Comlink from 'comlink'
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let session: ort.InferenceSession | null = null
let labels: string[] = ['吃飯', '我', '你', '有沒有']

const AIWorker = {
  async loadModel(modelUrl: string) {
    try {
      // 一次性獲取模型數據
      console.log("[Worker] 開始載入模型:", modelUrl);
      if (session) return true; // 已載入模型則直接返回

      const response = await fetch(modelUrl);
      if (!response.ok) throw new Error(`Fetch failed: ${response.status}`);
      const buffer = await response.arrayBuffer();

      ort.env.wasm.numThreads = 1; // 手機端建議設為 1 以免過熱

      // 初始化 Session
      session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      console.log('[Worker] 模型載入成功');
      return true;
    } catch (e) {
      console.error('[Worker] 初始化失敗', e);
      return false;
    }
  },

  async predict(landmarks: number[]) {
    if (!session) return '模型未載入'

    try {
      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      if (!inputName || !outputName) return '模型輸入輸出名稱異常'

      if (landmarks.length !== 3780) {
        return `資料長度不符，預期 3780，實際收到 ${landmarks.length}`;
      }
      const inputData = new Float32Array(landmarks);
      // 將 MediaPipe 傳來的節點轉換為 ONNX 需要的 Tensor (假設模型輸入為 1x63 的 Float32Array)
      const inputTensor = new ort.Tensor('float32', inputData, [1, 3780]);

      const feeds: Record<string, ort.Tensor> = {}
      feeds[inputName] = inputTensor

      const results = await session.run(feeds)
      const outputTensor = results[outputName]
      if (!outputTensor) return '模型輸出異常'

      const output = outputTensor.data as Float32Array

      if (output.length === 0) return '模型輸出為空'

      // 簡單的 ArgMax 找出機率最高的類別 (需對應你訓練的 10 個手語標籤)
      let maxIndex = 0;
      let maxScore:number = -Infinity;
      for (let i = 0; i < output.length; i++) {
        const score = output[i];
        if (score! > maxScore) {
          maxScore = score!;
          maxIndex = i;
        }
      }

      return labels[maxIndex] ?? '未知類別'
    } catch (e) {
      console.error('[Worker] 推論異常:', e);
      return `辨識錯誤，${e instanceof Error ? e.message : String(e)}`;
    }
  }
}
// 暴露 Worker API 給主線程使用
Comlink.expose(AIWorker)
export type AIWorkerType = typeof AIWorker