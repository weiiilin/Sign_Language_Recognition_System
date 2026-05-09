<template>
  <div class="overlay" @click.self="$emit('close')">
    <section class="sheet">
      <div class="handle"></div>

      <h2>翻譯細項</h2>

      <p class="blue">查詢：{{ props.word }}</p>

      <p>
        拆解：
        {{ detail?.breakdown?.join(' + ') }}
      </p>

      <p class="blue">手語分解動作</p>

      <p>
        手型：{{ detail?.handshape?.join(', ') }}
      </p>
      <img
        v-for="img in detail?.ＨandshapeImage"
        :key="img"
        :src="img"
        class="image"
      />
      <p>
        位置：{{ detail?.location?.join(', ') }}
      </p>
      <img
        v-for="img in detail?.positionImage"
        :key="img"
        :src="img"
        class="image"
      />

      <p>
        動作：{{ detail?.movement }}
      </p>
      <video
        v-if="detail?.video"
        :src="detail.video"
        controls
        class="video"
      />
      

    </section>
  </div>
</template>

<script setup>
import { signDictionary } from '../../data/signDictionary'

const props = defineProps({
  word: {
    type: String,
    default: ''
  }
})
const detail = computed(() => {
  return signDictionary[props.word] || null
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
.video {
  width: 100%;
  max-height: 220px;

  border-radius: 16px;

  object-fit: cover;

  margin-top: 12px;
}

.image {
  width: 100%;
  max-width: 140px;

  border-radius: 12px;

  object-fit: contain;

  margin-top: 10px;
}
.image-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>