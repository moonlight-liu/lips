<script setup lang="ts">
import type { TimelinePoint } from '../api/types'

const props = defineProps<{
  points: TimelinePoint[]
}>()

const width = 720
const height = 180
const padding = 18

function buildPath() {
  if (!props.points.length) return ''
  const maxIndex = Math.max(1, props.points.length - 1)
  return props.points
    .map((point, index) => {
      const x = padding + (index / maxIndex) * (width - padding * 2)
      const y = padding + (1 - point.score) * (height - padding * 2)
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')
}
</script>

<template>
  <div class="score-chart">
    <div class="chart-header">
      <span>Window score timeline</span>
      <span class="threshold">threshold 0.50</span>
    </div>

    <svg viewBox="0 0 720 180" role="img" aria-label="Detection score timeline">
      <line x1="18" x2="702" y1="90" y2="90" class="threshold-line" />
      <path :d="buildPath()" class="score-line" fill="none" />
      <circle
        v-for="(point, index) in points"
        :key="point.index"
        :cx="18 + (index / Math.max(1, points.length - 1)) * 684"
        :cy="18 + (1 - point.score) * 144"
        r="3.6"
        :class="point.score >= 0.5 ? 'fake-dot' : 'real-dot'"
      />
    </svg>
  </div>
</template>
