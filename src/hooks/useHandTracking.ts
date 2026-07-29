import { useCallback, useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'
import type {
  DebugSnapshot,
  TrackingStatus,
  WorkerResponse,
} from '../types/handTracking'

interface UseHandTrackingOptions {
  videoRef: RefObject<HTMLVideoElement | null>
}

const DEBUG_UPDATE_INTERVAL_MS = 100
const CONSOLE_UPDATE_INTERVAL_MS = 1_000

export function useHandTracking({ videoRef }: UseHandTrackingOptions) {
  const [status, setStatus] = useState<TrackingStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [result, setResult] = useState<HandLandmarkerResult | null>(null)
  const [debugSnapshot, setDebugSnapshot] = useState<DebugSnapshot | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const workerReadyRef = useRef(false)
  const shouldRunRef = useRef(false)
  const frameInFlightRef = useRef(false)
  const lastVideoTimeRef = useRef(-1)
  const lastTimestampRef = useRef(-1)
  const fpsWindowStartedAtRef = useRef(performance.now())
  const fpsFrameCountRef = useRef(0)
  const fpsRef = useRef(0)
  const lastDebugUpdateRef = useRef(0)
  const lastConsoleUpdateRef = useRef(0)

  const cancelLoop = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    frameInFlightRef.current = false
    lastVideoTimeRef.current = -1
  }, [])

  const startLoop = useCallback(() => {
    cancelLoop()

    const processFrame = async () => {
      animationFrameRef.current = requestAnimationFrame(processFrame)

      const video = videoRef.current
      const worker = workerRef.current

      if (
        !shouldRunRef.current ||
        !workerReadyRef.current ||
        !worker ||
        !video ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        frameInFlightRef.current ||
        video.currentTime === lastVideoTimeRef.current
      ) {
        return
      }

      frameInFlightRef.current = true
      lastVideoTimeRef.current = video.currentTime

      try {
        const bitmap = await createImageBitmap(video)

        if (!shouldRunRef.current || !workerRef.current) {
          bitmap.close()
          frameInFlightRef.current = false
          return
        }

        const now = performance.now()
        const timestampMs =
          now > lastTimestampRef.current ? now : lastTimestampRef.current + 1
        lastTimestampRef.current = timestampMs

        workerRef.current.postMessage(
          { type: 'DETECT', bitmap, timestampMs },
          [bitmap],
        )
      } catch (frameError) {
        frameInFlightRef.current = false
        console.error('Kamera karesi hazırlanamadı.', frameError)
      }
    }

    animationFrameRef.current = requestAnimationFrame(processFrame)
  }, [cancelLoop, videoRef])

  const stop = useCallback(() => {
    shouldRunRef.current = false
    cancelLoop()

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setCameraActive(false)
    setResult(null)
    setDebugSnapshot(null)
    setError(null)
    setStatus('stopped')
  }, [cancelLoop, videoRef])

  const start = useCallback(async () => {
    if (streamRef.current) {
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Bu tarayıcı kamera erişimini desteklemiyor.')
      setStatus('error')
      return
    }

    setError(null)
    setStatus('requesting-camera')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })

      const video = videoRef.current
      if (!video) {
        stream.getTracks().forEach((track) => track.stop())
        throw new Error('Video elementi hazırlanamadı.')
      }

      streamRef.current = stream
      video.srcObject = stream
      await video.play()

      setCameraActive(true)
      shouldRunRef.current = true
      if (workerReadyRef.current) {
        setStatus('running')
        startLoop()
      } else {
        setStatus('loading-model')
      }
    } catch (cameraError) {
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      setCameraActive(false)
      const message =
        cameraError instanceof Error
          ? cameraError.message
          : 'Kamera başlatılamadı.'
      setError(message)
      setStatus('error')
    }
  }, [startLoop, videoRef])

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/handLandmarker.worker.ts', import.meta.url),
      { type: 'module' },
    )
    workerRef.current = worker
    setStatus('loading-model')

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data

      if (message.type === 'READY') {
        workerReadyRef.current = true
        setStatus(shouldRunRef.current ? 'running' : 'ready')

        if (shouldRunRef.current && streamRef.current) {
          startLoop()
        }
        return
      }

      if (message.type === 'ERROR') {
        frameInFlightRef.current = false
        setError(message.message)
        setStatus('error')
        return
      }

      frameInFlightRef.current = false
      setResult(message.result)

      const now = performance.now()
      fpsFrameCountRef.current += 1
      const fpsWindowDuration = now - fpsWindowStartedAtRef.current

      if (fpsWindowDuration >= 500) {
        fpsRef.current =
          (fpsFrameCountRef.current * 1_000) / fpsWindowDuration
        fpsWindowStartedAtRef.current = now
        fpsFrameCountRef.current = 0
      }

      if (now - lastDebugUpdateRef.current >= DEBUG_UPDATE_INTERVAL_MS) {
        setDebugSnapshot({
          result: message.result,
          inferenceTimeMs: message.inferenceTimeMs,
          fps: fpsRef.current,
        })
        lastDebugUpdateRef.current = now
      }

      if (now - lastConsoleUpdateRef.current >= CONSOLE_UPDATE_INTERVAL_MS) {
        console.debug('[ChordFlow] Hand tracking', {
          hands: message.result.landmarks.length,
          handedness: message.result.handedness.map(
            (categories) => categories[0]?.categoryName ?? 'Unknown',
          ),
          inferenceTimeMs: Number(message.inferenceTimeMs.toFixed(1)),
          fps: Number(fpsRef.current.toFixed(1)),
        })
        lastConsoleUpdateRef.current = now
      }
    }

    worker.onerror = (workerError) => {
      frameInFlightRef.current = false
      setError(workerError.message || 'MediaPipe worker hatası.')
      setStatus('error')
    }

    worker.postMessage({ type: 'INITIALIZE' })

    return () => {
      shouldRunRef.current = false
      cancelLoop()
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      worker.terminate()
      workerRef.current = null
      workerReadyRef.current = false
    }
  }, [cancelLoop, startLoop])

  return {
    status,
    error,
    result,
    debugSnapshot,
    isRunning: status === 'running',
    cameraActive,
    start,
    stop,
  }
}
