import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import {
  DrawingUtils,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision'
import type { TrackingStatus } from '../types/handTracking'

interface CameraStageProps {
  videoRef: RefObject<HTMLVideoElement | null>
  result: HandLandmarkerResult | null
  status: TrackingStatus
  cameraActive: boolean
}

const STATUS_LABELS: Record<TrackingStatus, string> = {
  idle: 'Starting',
  'loading-model': 'Loading model',
  ready: 'Camera ready',
  'requesting-camera': 'Waiting for camera',
  running: 'Live',
  stopped: 'Camera stopped',
  error: 'Connection error',
}

export function CameraStage({
  videoRef,
  result,
  status,
  cameraActive,
}: CameraStageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) {
      return
    }

    const width = video.videoWidth || 1280
    const height = video.videoHeight || 720

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    context.clearRect(0, 0, canvas.width, canvas.height)

    if (!result) {
      return
    }

    const drawingUtils = new DrawingUtils(context)

    result.landmarks.forEach((landmarks) => {
      drawingUtils.drawConnectors(
        landmarks,
        HandLandmarker.HAND_CONNECTIONS,
        {
          color: '#8b5cf6',
          lineWidth: 4,
        },
      )
      drawingUtils.drawLandmarks(landmarks, {
        color: '#22d3ee',
        fillColor: '#07111f',
        lineWidth: 2,
        radius: 4,
      })
    })
  }, [result, videoRef])

  return (
    <section className="camera-card" aria-label="Camera and hand tracking">
      <div className="camera-stage">
        <video
          ref={videoRef}
          className="camera-video"
          playsInline
          muted
          aria-label="Live camera feed"
        />
        <canvas
          ref={canvasRef}
          className="landmark-canvas"
          aria-label="Detected hand landmarks"
        />

        {!cameraActive && (
          <div className="camera-placeholder">
            <div className="hand-mark" aria-hidden="true">
              ◇
            </div>
            <p>Open the camera to start hand tracking</p>
            <span>Video stays on this device.</span>
          </div>
        )}

        <div className={`live-badge live-badge--${status}`}>
          <span aria-hidden="true" />
          {STATUS_LABELS[status]}
        </div>
      </div>
    </section>
  )
}
