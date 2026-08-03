<template>
  <div class="overlay" @click.self="$emit('close')">
    <div class="modal">
      <h2>設定</h2>
      <p>翻譯設定</p>
      <hr />
      <div class="setting-row">
        <p>文字大小：{{ fontSize }}px</p>
        <input
          v-model="fontSize"
          type="range"
          min="12"
          max="24"
        />
      </div>

      <div class="setting-row">
        <p>每頁顯示詞彙：{{ dictionaryPageSize }} 個</p>
        <select v-model.number="dictionaryPageSize">
          <option v-for="option in pageSizeOptions" :key="option" :value="option">
            {{ option }}
          </option>
        </select>
      </div>

      <h3>關於</h3>
      <hr />
      <p class="small">
        台灣手語翻譯系統
        <br />
        Version 1.0
        <br />
        製作人：余俊霖、黃暐淋
        <br />
      </p>

    </div>
  </div>
</template>

<script setup>
import { useSignStore } from '@/stores/signStore'

const signStore = useSignStore()
const fontSize = ref(16)
const pageSizeOptions = [5, 10, 20, 50]
const dictionaryPageSize = computed({
  get() {
    return signStore.dictionaryPageSize
  },
  set(size) {
    signStore.setDictionaryPageSize(size)
  }
})

onMounted(() => {
  const savedSize = localStorage.getItem('fontSize')

  if (savedSize) {
    fontSize.value = Number(savedSize)
    document.documentElement.style.setProperty(
      '--app-font-size',
      `${fontSize.value}px`
    )
  }

  signStore.loadDictionaryPageSize()
})

watch(fontSize, (newSize) => {
  document.documentElement.style.setProperty(
    '--app-font-size',
    `${newSize}px`
  )
  localStorage.setItem('fontSize', String(newSize))
})
</script>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.3);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal {
  width: 260px;
  background: white;
  border-radius: 24px;
  padding: 20px;
}

.small {
  font-size: 12px;
}

.setting-row {
  margin: 16px 0;
}

.setting-row input {
  width: 100%;
}

.setting-row select {
  width: 100%;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 8px 10px;
  background: white;
}

button {
  margin-top: 12px;
}
</style>
