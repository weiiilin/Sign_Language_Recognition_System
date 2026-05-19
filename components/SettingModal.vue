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
const fontSize = ref(16)

onMounted(() => {
  const savedSize = localStorage.getItem('fontSize')

  if (savedSize) {
    fontSize.value = Number(savedSize)
    document.documentElement.style.setProperty(
      '--app-font-size',
      `${fontSize.value}px`
    )
  }
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

button {
  margin-top: 12px;
}
</style>