<template>
  <div class="overlay" @click.self="$emit('close')">
    <section class="sheet">
      <div class="handle"></div>

      <h2>翻譯細項</h2>

      <p class="blue">查詢：{{ word }}</p>
      <p>拆解 → {{ breakdown }}</p>

      <p class="blue">手語分解動作</p>
      <p v-html="detail"></p>
    </section>
  </div>
</template>

<script setup>
defineProps({
  word: {
    type: String,
    default: '尚未辨識'
  },
  breakdown: {
    type: String,
    default: '尚無拆解資料'
  },
  detail: {
    type: String,
    default: '尚無動作說明'
  }
})
</script>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.25);
  display: flex;
  align-items: flex-end;
  z-index: 50;
}
.sheet {
  width: 100%;
  min-height: 380px;
  background: white;
  border-radius: 28px 28px 0 0;
  padding: 24px;
}
.handle {
  width: 48px;
  height: 5px;
  background: #ccc;
  border-radius: 999px;
  margin: 0 auto 20px;
}
.blue {
  color: #1e8cff;
}
</style>