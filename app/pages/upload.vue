<template>
  <main class="page">
    <header class="header">
      <h1>手語翻譯LOGO</h1>
      <button @click="showSetting = true">☰</button>
    </header>
  <div class="content"></div>
    <section class="upload-box">
      <button class="upload-btn">上傳影片</button>
    </section>

    <div class="pill">
      台灣手語<span>→</span> 中文（繁體）
    </div>

    <section class="result-box" @click="showDetail = true">
      <button class="expand">↗</button>
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
  background: #f5f5fa;
  padding: 0 34px 120px;
}

.header {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;

  height:70px;
  background: #fbfbff;
  border-top: none;
  border-radius: 0 0 40px 40px;

  display: flex;
  align-items: center;
  justify-content: center;
  gap: 40px;

  box-shadow: 0 6px 8px rgba(0, 0, 0, 0.14);

  z-index: 100;
}

.header h1 {
  font-size: 28px;
  font-weight: 400;
}

.header button {
  border: none;
  background: transparent;
  font-size: 34px;
}

.content {
  flex: 1;
  overflow-y: auto;

  margin-top: 0px;  
  padding: 24px 34px 120px;
}

.upload-box {
  height: 356px;
  background: white;
  border-radius: 26px;
  margin-top: 72px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.upload-btn {
  border: none;
  background: #0d8cff;
  color: white;
  border-radius: 999px;
  padding: 8px 20px;
  font-size: 17px;
}

.pill {
  margin: 12px auto;
  display: flex;              
  align-items: center;
  justify-content: center;
  width: fit-content;
  background: #dedede;
  color: #0078df;
  border-radius: 999px;
  padding: 8px 28px;
  font-size: 17px;
}

.pill span {
  padding: 0 16px;
  font-size: 28px;
}

.result-box {
  height: 276px;
  background: white;
  border-radius: 34px;
  position: relative;
}

.expand {
  position: absolute;
  right: 22px;
  bottom: 22px;
  border: none;
  background: transparent;
  font-size: 34px;
}

.page {
  min-height: 100vh;
  background: #f5f5fa;
  padding: 0 34px 120px;

  padding-top: 180px;
}

@media (max-width: 430px) {
  .page {
    padding-left: 24px;
    padding-right: 24px;
  }

  .header {
    margin-left: -24px;
    margin-right: -24px;
  }
}
</style>