/// <reference lib="webworker" />

import {
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision'
import type {
  WorkerRequest,
  WorkerResponse,
} from '../types/handTracking'

const WASM_BASE_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.0/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

let handLandmarker: HandLandmarker | null = null

function respond(message: WorkerResponse) {
  self.postMessage(message)
}

async function createLandmarker(delegate: 'GPU' | 'CPU') {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL, true)

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
}

async function initialize() {
  if (handLandmarker) {
    respond({ type: 'READY' })
    return
  }

  try {
    try {
      handLandmarker = await createLandmarker('GPU')
    } catch (gpuError) {
      console.warn('GPU delegate kullanılamadı, CPU deneniyor.', gpuError)
      handLandmarker = await createLandmarker('CPU')
    }

    respond({ type: 'READY' })
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'MediaPipe modeli yüklenemedi.'
    respond({ type: 'ERROR', message })
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data

  if (message.type === 'INITIALIZE') {
    await initialize()
    return
  }

  if (!handLandmarker) {
    message.bitmap.close()
    respond({ type: 'ERROR', message: 'Hand Landmarker henüz hazır değil.' })
    return
  }

  const startedAt = performance.now()

  try {
    const result = handLandmarker.detectForVideo(
      message.bitmap,
      message.timestampMs,
    )

    respond({
      type: 'RESULT',
      result,
      inferenceTimeMs: performance.now() - startedAt,
    })
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : 'El algılama başarısız oldu.'
    respond({ type: 'ERROR', message: errorMessage })
  } finally {
    message.bitmap.close()
  }
}
