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
  idle: 'Başlatılıyor',
  'loading-model': 'Model yükleniyor',
  ready: 'Kamera hazır',
  'requesting-camera': 'Kamera izni bekleniyor',
  running: 'Canlı takip',
  stopped: 'Kamera durduruldu',
  error: 'Bağlantı hatası',
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
    <section className="camera-card" aria-label="Kamera ve el takip alanı">
      <div className="camera-stage">
        <video
          ref={videoRef}
          className="camera-video"
          playsInline
          muted
          aria-label="Canlı kamera görüntüsü"
        />
        <canvas
          ref={canvasRef}
          className="landmark-canvas"
          aria-label="Algılanan el noktaları"
        />

        {!cameraActive && (
          <div className="camera-placeholder">
            <div className="hand-mark" aria-hidden="true">
              ◇
            </div>
            <p>Kamerayı açarak el takibini başlat</p>
            <span>Görüntü yalnızca bu cihazda işlenir.</span>
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
