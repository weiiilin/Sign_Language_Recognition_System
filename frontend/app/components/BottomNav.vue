<template>
  <nav class="nav">
    <NuxtLink to="/" class="item">字典</NuxtLink>
    <NuxtLink to="/upload" class="item">上傳</NuxtLink>
    <NuxtLink to="/record" class="item">錄影</NuxtLink>
  </nav>
</template>

<style scoped>
.nav {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  padding: 12px 20px;
  border-radius: 999px;
  display: flex;
  gap: 20px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.item {
  text-decoration: none;
  color: #555;
}

.router-link-active {
  color: #1e8cff;
  font-weight: bold;
}
</style>