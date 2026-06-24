import type { DetectionResult, ModelOption } from './types'

const API_BASE = ''

export async function fetchModels(): Promise<ModelOption[]> {
  const response = await fetch(`${API_BASE}/api/models`)
  if (!response.ok) {
    throw new Error('Failed to load models')
  }
  const data = await response.json()
  return data.models
}

export async function runMockDetection(file: File, modelId: string): Promise<DetectionResult> {
  const body = new FormData()
  body.append('file', file)

  const response = await fetch(`${API_BASE}/api/detect/mock?model_id=${encodeURIComponent(modelId)}`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    throw new Error('Detection request failed')
  }

  return response.json()
}

export async function runRealDetection(
  file: File,
  audioFile: File | null,
  modelId: string,
): Promise<DetectionResult> {
  const body = new FormData()
  body.append('file', file)
  if (audioFile) {
    body.append('audio_file', audioFile)
  }

  const response = await fetch(`${API_BASE}/api/detect?model_id=${encodeURIComponent(modelId)}`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    let message = 'Detection request failed'
    try {
      const payload = await response.json()
      message = payload.detail ?? message
    } catch {
      // Keep the generic message when the backend did not return JSON.
    }
    throw new Error(message)
  }

  return response.json()
}
