<script setup>
import BottomNav from '../components/BottomNav.vue'
import SettingModal from '../components/SettingModal.vue'
import DetailSheet from '../components/DetailSheet.vue'

const showSetting = ref(false)
const showDetail = ref(false)
</script>

<template>
  <main class="page">

    <header class="top">
      <button @click="showSetting = true">☰</button>
      <input placeholder="搜尋" />
    </header>

    <section class="video">
      <div class="play">▶</div>
    </section>

    <div class="pill">
      中文（繁體） → 台灣手語
    </div>

    <section class="card" @click="showDetail = true">
      <p class="main">不好吃</p>
      <span>吃 + 爛。</span>
    </section>

    <BottomNav />

    <SettingModal
      v-if="showSetting"
      @close="showSetting = false"
    />

    <DetailSheet
      v-if="showDetail"
      @close="showDetail = false"
    />

  </main>
</template>

<style scoped>
.page {
  min-height: 100vh;
  background: #f6f7fb;
  padding: 24px;
}

.top {
  display: flex;
  gap: 10px;
}

.top input {
  flex: 1;
  border: none;
  border-radius: 999px;
  padding: 10px;
}

.video {
  margin-top: 20px;
  height: 180px;
  background: #bbb;
  border-radius: 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
}

.play {
  background: #1e8cff;
  padding: 8px 16px;
  border-radius: 999px;
}

.pill {
  margin: 12px auto;
  background: #d8ecff;
  color: #2488e8;
  padding: 6px 12px;
  border-radius: 999px;
  width: fit-content;
}

.card {
  margin-top: 16px;
  background: white;
  border-radius: 20px;
  padding: 20px;
}

.main {
  font-size: 20px;
}
</style>