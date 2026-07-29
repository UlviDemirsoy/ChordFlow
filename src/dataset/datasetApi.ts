import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

export type DegreeLabel = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
export type DatasetHand = 'left' | 'right'

export interface DatasetCounts {
  counts: Record<string, number>
  total: number
}

interface SaveSampleInput {
  degree: DegreeLabel
  imageDataUrl: string
  landmarks: NormalizedLandmark[]
  handednessScore: number
}

export async function getDatasetCounts(
  hand: DatasetHand = 'left',
): Promise<DatasetCounts> {
  const response = await fetch(`/api/dataset/${hand}`)
  if (!response.ok) {
    throw new Error('Dataset sayaçları okunamadı.')
  }
  return response.json() as Promise<DatasetCounts>
}

export async function saveHandSample(
  hand: DatasetHand,
  input: SaveSampleInput,
) {
  const response = await fetch(`/api/dataset/${hand}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      degree: input.degree,
      imageDataUrl: input.imageDataUrl,
      capturedAt: new Date().toISOString(),
      handednessScore: input.handednessScore,
      landmarks: input.landmarks.map(({ x, y, z }) => ({ x, y, z })),
    }),
  })

  const payload = (await response.json()) as {
    id?: string
    imagePath?: string
    error?: string
  }

  if (!response.ok) {
    throw new Error(payload.error || 'Örnek kaydedilemedi.')
  }

  return payload as { id: string; imagePath: string }
}

export function saveLeftHandSample(input: SaveSampleInput) {
  return saveHandSample('left', input)
}

export function saveRightHandSample(input: SaveSampleInput) {
  return saveHandSample('right', input)
}

export function captureVideoAsJpeg(video: HTMLVideoElement) {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error('Kamera görüntüsü henüz hazır değil.')
  }

  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Görüntü yakalama canvası oluşturulamadı.')
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/jpeg', 0.9)
}
