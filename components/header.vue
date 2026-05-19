<template>
    <header class="top-bar flex items-center justify-between w-full relative mb-6">
        <button class="setting flex-shrink-0 transition-transform active:scale-95" @click="signStore.openSetting()">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2"
                stroke="currentColor" class="w-6 h-6 text-gray-700">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
            </svg>
        </button>

        <div class="flex-1 flex items-center justify-between ml-4">
            <slot />
        </div>
    </header>
</template>

<script setup lang="ts">
import { useSignStore } from '@/stores/signStore'
const signStore = useSignStore()
</script>