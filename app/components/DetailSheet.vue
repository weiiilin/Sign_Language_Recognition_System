<template>
  <div class="overlay" @click.self="$emit('close')">
    <section class="sheet">
      <div class="handle"></div>
      <h2>翻譯細項</h2>
      <p class="blue">查詢：不好吃</p>
      <p>拆解 → 吃 + 爛。</p>
      <p class="blue">手語分解動作</p>
      <p>手型：B 手型<br />位置：胸前<br />動作：向外移動</p>
    </section>
  </div>
</template>

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