<template>
  <div class="overlay" @click.self="$emit('close')">
    <div class="modal">
      <h2>設定</h2>

      <p>翻譯設定</p>
      <hr />

      <p>文字大小</p>
      <input type="range" />

      <h3>關於</h3>
      <hr />

      <p class="small">
        台灣手語翻譯系統<br />
        Version 1.0
      </p>

      <button @click="$emit('close')">關閉</button>
    </div>
  </div>
</template>

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

button {
  margin-top: 12px;
}
</style>