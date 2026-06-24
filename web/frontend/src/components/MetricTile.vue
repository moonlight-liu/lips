<script setup lang="ts">
defineProps<{
  label: string
  value: string
  tone?: 'neutral' | 'good' | 'alert'
}>()
</script>

<template>
  <section class="metric-tile" :class="tone ?? 'neutral'">
    <span>{{ label }}</span>
    <strong>{{ value }}</strong>
  </section>
</template>
