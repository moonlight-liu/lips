<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  Activity,
  AudioWaveform,
  BadgeCheck,
  BarChart3,
  Clock3,
  Cpu,
  FileVideo,
  Layers3,
  Play,
  Radar,
  ShieldCheck,
  UploadCloud,
} from 'lucide-vue-next'

import { fetchModels, runMockDetection, runRealDetection } from './api/client'
import type { DetectionResult, ModelOption, TimelinePoint } from './api/types'
import ScoreChart from './components/ScoreChart.vue'

type PipelineMode = 'real' | 'mock'

const models = ref<ModelOption[]>([])
const selectedModel = ref('lipfd-light-best')
const selectedFile = ref<File | null>(null)
const selectedAudioFile = ref<File | null>(null)
const pipelineMode = ref<PipelineMode>('real')
const videoUrl = ref('')
const result = ref<DetectionResult | null>(null)
const loading = ref(false)
const error = ref('')
const videoDragActive = ref(false)
const audioDragActive = ref(false)

const probabilityPercent = computed(() => {
  if (!result.value) return 0
  return Math.round(result.value.fakeProbability * 1000) / 10
})

const verdictTone = computed(() => {
  if (!result.value) return 'idle'
  return result.value.label === 'fake' ? 'attention' : 'clear'
})

const riskSegments = computed(() => {
  const points = result.value?.timeline ?? []
  return points
    .filter((point) => point.score >= 0.5)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
})

const statusLabel = computed(() => {
  if (loading.value) return '分析中'
  if (result.value) return result.value.mode === 'mock' ? '模拟结果' : '真实模型结果'
  return '就绪'
})

const modelLabel = computed(() => {
  if (pipelineMode.value === 'mock') return '模拟流程'
  if (result.value?.backend?.residentModel) return '常驻模型'
  return '真实模型'
})

onMounted(async () => {
  try {
    models.value = await fetchModels()
  } catch {
    models.value = [
      {
        id: 'lipfd-light-best',
        name: 'LipFD-Light 最佳模型',
        description: 'ViT-B/32 + ResNet18 常驻模型。',
        status: 'ready',
      },
    ]
  }
})

function isVideoFile(file: File) {
  return file.type.startsWith('video/') || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name)
}

function isAudioFile(file: File) {
  return file.type.startsWith('audio/') || /\.(wav|mp3|m4a|aac|flac)$/i.test(file.name)
}

function setVideoFile(file: File) {
  selectedFile.value = file
  result.value = null
  error.value = ''

  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  videoUrl.value = URL.createObjectURL(file)
}

function setAudioFile(file: File) {
  selectedAudioFile.value = file
  result.value = null
  error.value = ''
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  if (!isVideoFile(file)) {
    error.value = '请拖入或选择视频文件。'
    return
  }
  setVideoFile(file)
}

function onAudioFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  if (!isAudioFile(file)) {
    error.value = '请拖入或选择音频文件。'
    return
  }
  setAudioFile(file)
}

function onVideoDrop(event: DragEvent) {
  videoDragActive.value = false
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  if (!isVideoFile(file)) {
    error.value = '拖入的不是视频文件，请选择 mp4、mov、avi 或 webm。'
    return
  }
  setVideoFile(file)
}

function onAudioDrop(event: DragEvent) {
  audioDragActive.value = false
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  if (!isAudioFile(file)) {
    error.value = '拖入的不是音频文件，请选择 WAV 或常见音频格式。'
    return
  }
  setAudioFile(file)
}

function formatFileSize(file: File | null) {
  if (!file) return '--'
  return `${(file.size / (1024 * 1024)).toFixed(2)} MB`
}

function formatSeconds(value?: number) {
  if (value === undefined || Number.isNaN(value)) return '--'
  return `${value.toFixed(2)}s`
}

function formatMs(value?: number) {
  if (value === undefined || value === null || Number.isNaN(value)) return '--'
  return `${value.toFixed(1)} ms`
}

const runtimeBreakdown = computed(() => {
  const backend = result.value?.backend
  const stage = backend?.preprocessStageMsPerWindow ?? {}
  return [
    ['请求总耗时', formatMs(backend?.requestElapsedMs)],
    ['检测链路', formatMs(backend?.detectElapsedMs)],
    ['音频抽取', formatMs(backend?.audioExtractMs)],
    ['视频解码/窗口采样', formatMs(backend?.videoDecodeMsPerVideo)],
    ['音频 Mel', formatMs(backend?.audioMelMsPerVideo)],
    ['帧张量缓存', formatMs(stage.frame_tensor_cache)],
    ['Mel 张量批处理', formatMs(stage.top_mel_tensor_batch)],
    ['GPU 预处理同步', formatMs(stage.preprocess_device_sync)],
    ['模型前向', formatMs(backend?.transferForwardMsPerWindow)],
  ]
})

function segmentLabel(point: TimelinePoint) {
  const start = point.startTime ?? point.time
  const end = point.endTime ?? point.time + 0.5
  return `${formatSeconds(start)} - ${formatSeconds(end)}`
}

function localizeError(message: string) {
  if (message.includes('音频') || message.includes('audio')) {
    return '无法读取视频音频轨。请确认视频包含声音，或上传对应 WAV 文件。'
  }
  if (message.includes('过短') || message.includes('too short')) return message
  if (message.includes('超时') || message.includes('timed out')) return '检测超时。请裁剪视频后重试。'
  if (message.includes('Failed to fetch')) return '无法连接后端服务。请确认 8000 端口已启动。'
  return message || '检测失败，请查看后端日志。'
}

async function startDetection() {
  if (!selectedFile.value) {
    error.value = '请先选择视频文件。'
    return
  }

  loading.value = true
  error.value = ''
  result.value = null

  try {
    result.value = pipelineMode.value === 'mock'
      ? await runMockDetection(selectedFile.value, selectedModel.value)
      : await runRealDetection(selectedFile.value, selectedAudioFile.value, selectedModel.value)
  } catch (err) {
    error.value = localizeError(err instanceof Error ? err.message : '')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div class="identity">
          <span>LipFD-Light</span>
        <h1>视频真伪分析</h1>
      </div>
      <div class="system-pill">
        <span :class="{ active: loading }"></span>
        {{ statusLabel }}
      </div>
    </header>

    <section class="workspace">
      <aside class="panel controls-panel">
        <div class="panel-heading">
          <Radar :size="18" />
          <span>输入</span>
        </div>

        <label
          class="upload-card primary-upload"
          :class="{ 'drag-active': videoDragActive }"
          @dragenter.prevent="videoDragActive = true"
          @dragover.prevent="videoDragActive = true"
          @dragleave.prevent="videoDragActive = false"
          @drop.prevent="onVideoDrop"
        >
          <UploadCloud :size="28" />
          <strong>{{ selectedFile?.name ?? '选择视频' }}</strong>
          <span>点击选择，或直接将视频拖入这里</span>
          <input type="file" accept="video/*" @change="onFileChange" />
        </label>

        <label
          class="upload-card subtle-upload"
          :class="{ 'drag-active': audioDragActive }"
          @dragenter.prevent="audioDragActive = true"
          @dragover.prevent="audioDragActive = true"
          @dragleave.prevent="audioDragActive = false"
          @drop.prevent="onAudioDrop"
        >
          <AudioWaveform :size="20" />
          <strong>{{ selectedAudioFile?.name ?? '可选 WAV 音频' }}</strong>
          <span>点击选择，或将音频文件拖入这里</span>
          <input type="file" accept="audio/wav,.wav" @change="onAudioFileChange" />
        </label>

        <div class="form-row">
          <label>检测流程</label>
          <div class="segmented">
            <button type="button" :class="{ active: pipelineMode === 'real' }" @click="pipelineMode = 'real'">
              真实模型
            </button>
            <button type="button" :class="{ active: pipelineMode === 'mock' }" @click="pipelineMode = 'mock'">
              模拟
            </button>
          </div>
        </div>

        <div class="form-row">
          <label for="model">模型</label>
          <select id="model" v-model="selectedModel">
            <option v-for="model in models" :key="model.id" :value="model.id">
              {{ model.name }}
            </option>
          </select>
        </div>

        <button class="primary-action" type="button" :disabled="loading" @click="startDetection">
          <Play :size="17" />
          <span>{{ loading ? '正在分析...' : '开始分析' }}</span>
        </button>

        <p v-if="error" class="error-note">{{ error }}</p>

        <div class="compact-facts">
          <div>
            <span>视频</span>
            <strong>{{ formatFileSize(selectedFile) }}</strong>
          </div>
          <div>
            <span>音频</span>
            <strong>{{ selectedAudioFile ? '已提供' : '自动抽取' }}</strong>
          </div>
        </div>
      </aside>

      <section class="panel media-panel">
        <div class="media-header">
          <div>
            <span>预览</span>
            <strong>{{ selectedFile?.name ?? '尚未选择文件' }}</strong>
          </div>
          <em>{{ modelLabel }}</em>
        </div>

        <div class="video-frame">
          <video v-if="videoUrl" :src="videoUrl" controls />
          <div v-else class="empty-preview">
            <FileVideo :size="34" />
            <span>等待上传视频</span>
          </div>
        </div>

        <div class="media-metrics">
          <div>
            <span>帧率</span>
            <strong>{{ result?.video ? result.video.fps : '--' }}</strong>
          </div>
          <div>
            <span>帧数</span>
            <strong>{{ result?.video ? result.video.frameCount : '--' }}</strong>
          </div>
          <div>
            <span>时长</span>
            <strong>{{ result?.video ? formatSeconds(result.video.durationSec) : '--' }}</strong>
          </div>
          <div>
            <span>窗口</span>
            <strong>{{ result?.metrics.windows ?? '--' }}</strong>
          </div>
        </div>
      </section>

      <aside class="panel verdict-panel" :class="verdictTone">
        <div class="panel-heading">
          <ShieldCheck :size="18" />
          <span>评估</span>
        </div>

        <div class="verdict-copy">
          <strong>{{ result?.label === 'fake' ? '可疑' : result?.label === 'real' ? '可信' : '待检测' }}</strong>
          <span>伪造概率</span>
        </div>

        <div class="probability-meter">
          <svg viewBox="0 0 120 120" aria-hidden="true">
            <circle cx="60" cy="60" r="52" class="meter-base" />
            <circle
              cx="60"
              cy="60"
              r="52"
              class="meter-value"
              :style="{ strokeDashoffset: 327 - (327 * probabilityPercent) / 100 }"
            />
          </svg>
          <span>{{ probabilityPercent.toFixed(1) }}%</span>
        </div>

        <div class="stat-list">
          <section>
            <Clock3 :size="15" />
            <span>延迟</span>
            <strong>{{ result ? formatMs(result.metrics.latencyMs) : '--' }}</strong>
          </section>
          <section>
            <BarChart3 :size="15" />
            <span>吞吐</span>
            <strong>{{ result ? `${result.metrics.fps}` : '--' }}</strong>
          </section>
          <section>
            <Cpu :size="15" />
            <span>运行方式</span>
            <strong>{{ result?.backend?.residentModel ? '常驻' : '--' }}</strong>
          </section>
        </div>
      </aside>
    </section>

    <section v-if="!result" class="idle-grid">
      <article class="panel idle-card">
        <Layers3 :size="18" />
        <div>
          <strong>上传视频</strong>
          <span>系统会读取帧率、时长和音频轨。</span>
        </div>
      </article>
      <article class="panel idle-card">
        <Activity :size="18" />
        <div>
          <strong>常驻模型分析</strong>
          <span>FastAPI 启动后复用 ViT-B/32 + ResNet18。</span>
        </div>
      </article>
      <article class="panel idle-card">
        <ShieldCheck :size="18" />
        <div>
          <strong>查看风险片段</strong>
          <span>检测完成后展示时间段、分数和耗时拆分。</span>
        </div>
      </article>
    </section>

    <section v-else class="lower-grid">
      <article class="panel chart-panel">
        <div class="section-title">
          <div>
            <span>时间轴</span>
            <strong>窗口置信度</strong>
          </div>
          <em>阈值 0.50</em>
        </div>
        <ScoreChart :points="result?.timeline ?? []" />
      </article>

      <article class="panel">
        <div class="section-title">
          <div>
            <span>片段</span>
            <strong>高风险片段</strong>
          </div>
          <em>{{ riskSegments.length }} 个</em>
        </div>
        <div v-if="riskSegments.length" class="risk-list">
          <section v-for="point in riskSegments" :key="point.index">
            <BadgeCheck :size="15" />
            <div>
              <strong>{{ segmentLabel(point) }}</strong>
              <span>窗口 {{ point.index }} · 分数 {{ point.score.toFixed(4) }}</span>
            </div>
          </section>
        </div>
        <p v-else class="empty-note">暂无超过阈值的片段</p>
      </article>

      <article class="panel">
        <div class="section-title">
          <div>
            <span>耗时</span>
            <strong>处理拆分</strong>
          </div>
          <em>{{ result?.backend?.preprocessDevice ?? '--' }}</em>
        </div>
        <dl class="runtime-list">
          <div v-for="[label, value] in runtimeBreakdown" :key="label">
            <dt>{{ label }}</dt>
            <dd>{{ value }}</dd>
          </div>
        </dl>
      </article>

      <article class="panel">
        <div class="section-title">
          <div>
            <span>兼容性</span>
            <strong>上传提示</strong>
          </div>
          <em>{{ result?.warnings?.length ?? 0 }}</em>
        </div>
        <div v-if="result?.warnings?.length" class="warning-list">
          <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
        </div>
        <p v-else class="empty-note">暂无兼容性提示</p>
      </article>
    </section>
  </main>
</template>
