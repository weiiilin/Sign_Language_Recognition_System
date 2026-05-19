<template>
  <main class="page max-w-2xl mx-auto flex flex-col justify-between min-h-screen pt-4 sm:pt-6 px-4 sm:px-6 pb-24">

    <AppHeader>
      <div class="logo text-center font-bold text-gray-800 flex-1">
        手語翻譯LOGO
      </div>
      <div class="w-12 h-12 flex-shrink-0"></div>
    </AppHeader>

    <section
      class="w-full aspect-video max-h-[300px] bg-white rounded-3xl flex flex-col justify-center items-center relative overflow-hidden shadow-sm border border-gray-100 my-2">
      <input ref="fileInput" type="file" accept="video/*" hidden @change="handleVideoUpload" />

      <div v-if="videoUrl" class="w-full h-full relative group">
        <video :src="videoUrl" controls playsinline class="w-full h-full object-contain bg-black" />
        <button @click="fileInput?.click()"
          class="absolute top-3 right-3 bg-black/60 hover:bg-black/80 text-white text-xs font-medium px-3 py-1.5 rounded-full backdrop-blur transition-all opacity-90 sm:opacity-0 group-hover:opacity-100">
          🔄 更換影片
        </button>
      </div>

      <div v-else @click="fileInput?.click()"
        class="w-full h-full border-2 border-dashed border-gray-200 hover:border-blue-400 m-2 rounded-2xl flex flex-col justify-center items-center p-6 cursor-pointer transition-colors group">
        <div class="text-4xl mb-3 transform group-hover:-translate-y-1 transition-transform duration-200">📤</div>
        <p class="text-sm font-medium text-gray-500 mb-4 text-center">
          點擊此處上傳手語錄影檔案<br />
          <span class="text-xs text-gray-400 font-normal">(支援所有常見影片格式)</span>
        </p>
        <button
          class="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm rounded-xl shadow-sm transition-colors pointer-events-none">
          選擇檔案
        </button>
      </div>
    </section>

    <div
      class="pill my-3 px-4 py-1.5 bg-blue-50 text-blue-600 rounded-full font-medium text-sm sm:text-base border border-blue-100 shadow-sm">
      台灣手語 <span class="mx-1">→</span> 中文（繁體）
    </div>

    <section
      class="w-full bg-white border border-gray-100 p-6 rounded-3xl shadow-sm cursor-pointer hover:shadow-md transition-shadow relative flex flex-col justify-between min-h-[140px] my-2 group"
      @click="signStore.openDetail({ word: videoUrl ? '影片分析中' : '尚未上傳', breakdown: '離線檔案排隊推論中' })">
      <div>
        <span class="text-xs font-bold text-blue-600 tracking-wider uppercase block mb-1">離線檔案分析結果</span>
        <p class="text-xl font-bold text-gray-700 group-hover:text-blue-600 transition-colors">
          {{ videoUrl ? '影片分析中，點擊查看詳細進度...' : '請先上傳影片以進行辨識' }}
        </p>
        <button
          class="absolute right-5 bottom-5 w-10 h-10 rounded-full bg-gray-50 flex items-center justify-center shadow-sm border border-gray-100">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5"
            stroke="currentColor" class="w-5 h-5 text-gray-500">
            <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 19.5 15-15m0 0H8.25m11.25 0v11.25" />
          </svg>
        </button>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useSignStore } from '@/stores/signStore'
import AppHeader from '@/components/header.vue'

const signStore = useSignStore()
const fileInput = ref<HTMLInputElement | null>(null)
const videoUrl = ref('')

const handleVideoUpload = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  videoUrl.value = URL.createObjectURL(file)
}
</script>