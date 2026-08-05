import { useEffect, useRef, useState } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'

const SMOOTHING_ALPHA = 0.18
const OUTPUT_DEADBAND = 0.015

/** Palm roll from index MCP ↔ pinky MCP; 0 = level, 1 = fully rolled. */
function getRightPalmTilt(result: HandLandmarkerResult | null) {
  if (!result) {
    return null
  }

  const index = result.handedness.findIndex(
    (categories) => categories[0]?.categoryName.toLowerCase() === 'right',
  )
  if (index < 0) {
    return null
  }

  const landmarks = result.landmarks[index]
  const indexMcp = landmarks?.[5]
  const pinkyMcp = landmarks?.[17]
  if (!indexMcp || !pinkyMcp) {
    return null
  }

  const dx = indexMcp.x - pinkyMcp.x
  const dy = indexMcp.y - pinkyMcp.y
  const span = Math.hypot(dx, dy)
  if (span < 1e-4) {
    return null
  }

  // |sin| of palm cross-vector vs horizontal → 0 flat, 1 edge-on
  return Math.min(1, Math.abs(dy) / span)
}

export function useRightHandTilt(result: HandLandmarkerResult | null) {
  const smoothedTiltRef = useRef<number | null>(null)
  const emittedTiltRef = useRef(0)
  const [tilt, setTilt] = useState(0)

  useEffect(() => {
    const raw = getRightPalmTilt(result)
    if (raw === null) {
      return
    }

    const previous = smoothedTiltRef.current
    const next =
      previous === null ? raw : previous + SMOOTHING_ALPHA * (raw - previous)
    smoothedTiltRef.current = next

    if (Math.abs(next - emittedTiltRef.current) >= OUTPUT_DEADBAND) {
      emittedTiltRef.current = next
      setTilt(next)
    }
  }, [result])

  return { tilt }
}
