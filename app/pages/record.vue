<template>
  <main class="page">

    <section class="camera">
      <button class="setting" @click="showSetting = true">☰</button>
      <button class="shutter">◎</button>
    </section>

    <div class="pill">
      台灣手語 → 中文（繁體）
    </div>

    <section class="card" @click="showDetail = true">
      <p>即時翻譯結果</p>
      <span>點擊查看細部內容</span>
    </section>

    <BottomNav />

    <SettingModal v-if="showSetting" @close="showSetting = false" />
    <DetailSheet v-if="showDetail" @close="showDetail = false" />

  </main>
</template>

<script setup>
const showSetting = ref(false)
const showDetail = ref(false)
</script>

<style scoped>
.page {
  min-height: 100vh;
  background: black;
  padding-bottom: 110px;
}

.camera {
  height: 65vh;
  background: white;
  border-radius: 0 0 28px 28px;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-end;
  padding-bottom: 40px;
}

.setting {
  position: absolute;
  top: 20px;
  left: 20px;
  border: none;
  background: #eee;
  border-radius: 14px;
  padding: 10px;
}

.shutter {
  border: none;
  background: #1e8cff;
  color: white;
  font-size: 32px;
  border-radius: 50%;
  width: 70px;
  height: 70px;
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
  background: white;
  border-radius: 20px;
  padding: 20px;
  margin: 16px 20px;
}

.card span {
  color: #1e8cff;
}
</style>
