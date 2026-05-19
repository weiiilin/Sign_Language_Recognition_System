<template>
  <main class="page max-w-2xl mx-auto flex flex-col justify-between min-h-screen pt-4 sm:pt-6 px-4 sm:px-6 pb-24">

    <AppHeader>
      <div class="relative flex-1">
        <input type="text" placeholder="搜尋手語單字、詞彙..." style=" position: relative;"
          class="w-full bg-white border border-gray-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 outline-none rounded-full py-3 px-5 text-sm shadow-sm transition-all text-gray-700 placeholder-gray-400" />
        <span class="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 text-sm">🔍</span>
      </div>
    </AppHeader>

    <section
      class="video-container w-full aspect-video max-h-[260px] bg-gray-300 rounded-3xl flex flex-col justify-center items-center text-white relative overflow-hidden shadow-inner group cursor-pointer border border-gray-100 my-2">
      <div
        class="play bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2.5 rounded-full shadow-md flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
          <path fill-rule="evenodd"
            d="M4.5 5.653c0-1.427 1.529-2.33 2.779-1.643l11.54 6.347c1.295.712 1.295 2.573 0 3.286L7.28 19.99c-1.25.687-2.779-.217-2.779-1.643V5.653Z"
            clip-rule="evenodd" />
        </svg>
        播放示範
      </div>
    </section>

    <div
      class="pill my-3 px-4 py-1.5 bg-blue-50 text-blue-600 rounded-full font-medium text-sm sm:text-base border border-blue-100 shadow-sm">
      中文（繁體） → 台灣手語
    </div>

    <section
      class="w-full bg-white border border-gray-100 p-5 rounded-2xl shadow-sm cursor-pointer hover:shadow-md transition-shadow mt-2 flex flex-col gap-1.5"
      @click="signStore.openDetail({ word: '不好吃', breakdown: '手語拆解：吃 + 爛。' })">
      <p class="text-2xl font-bold text-gray-800 tracking-wide">不好吃</p>
      <span class="text-sm text-gray-500 bg-gray-50 px-2.5 py-1 rounded-md w-fit font-medium">
        🖐️ 手語拆解：吃 + 爛。
      </span>
    </section>

  </main>
</template>

<script setup lang="ts">
import { useSignStore } from '@/stores/signStore'
import AppHeader from '@/components/header.vue'
const signStore = useSignStore()
</script>