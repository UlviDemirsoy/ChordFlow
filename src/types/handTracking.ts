import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'

export type TrackingStatus =
  | 'idle'
  | 'loading-model'
  | 'ready'
  | 'requesting-camera'
  | 'running'
  | 'stopped'
  | 'error'

export interface TrackingMetrics {
  fps: number
  inferenceTimeMs: number
}

export interface DebugSnapshot extends TrackingMetrics {
  result: HandLandmarkerResult
}

export type WorkerRequest =
  | { type: 'INITIALIZE' }
  | { type: 'DETECT'; bitmap: ImageBitmap; timestampMs: number }

export type WorkerResponse =
  | { type: 'READY' }
  | {
      type: 'RESULT'
      result: HandLandmarkerResult
      inferenceTimeMs: number
    }
  | { type: 'ERROR'; message: string }
