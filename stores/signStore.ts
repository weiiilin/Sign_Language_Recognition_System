// @ts-ignore
import { defineStore } from 'pinia'

export const useSignStore = defineStore('sign', {
  state: () => ({
    currentSign: '等待辨識中...' as string,
    isModelLoaded: false as boolean,
    fps: 0 as number,
    errorMsg: '' as string
  }),
  actions: {
    updateSign(this: any, sign: string) {
      this.currentSign = sign
    },
    setModelLoaded(this: any, status: boolean) {
      this.isModelLoaded = status
    },
    setError(this: any, msg: string) {
      this.errorMsg = msg
    }
  }
})