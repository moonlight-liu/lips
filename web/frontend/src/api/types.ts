export interface ModelOption {
  id: string
  name: string
  description: string
  status: string
}

export interface TimelinePoint {
  index: number
  time: number
  startTime?: number
  endTime?: number
  score: number
  label: 'real' | 'fake'
}

export interface DetectionResult {
  mode: string
  filename: string
  modelId: string
  label: 'real' | 'fake'
  fakeProbability: number
  confidence: number
  metrics: {
    latencyMs: number
    fps: number
    windows: number
    fileSizeMb: number
  }
  timeline: TimelinePoint[]
  warnings?: string[]
  video?: {
    frameCount: number
    fps: number
    width: number
    height: number
    durationSec: number
    sampledWindows: number
  }
  backend?: {
    residentModel?: boolean
    preprocessDevice?: string
    clip?: string
    backbone?: string
    totalSeconds?: number
    preModelMsPerWindow?: number
    transferForwardMsPerWindow?: number
    videoDecodeMsPerVideo?: number
    audioMelMsPerVideo?: number
    preprocessStageMsPerWindow?: Record<string, number>
    preprocessDetailMsPerWindow?: Record<string, number>
    uploadSaveMs?: number
    audioExtractMs?: number
    detectElapsedMs?: number
    requestElapsedMs?: number
  }
  createdAt: string
}
