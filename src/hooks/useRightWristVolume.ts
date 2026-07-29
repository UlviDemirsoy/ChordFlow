import { useEffect, useRef, useState } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'

const SMOOTHING_ALPHA = 0.16
const OUTPUT_DEADBAND = 0.012

function getRightWristY(result: HandLandmarkerResult | null) {
  if (!result) {
    return null
  }

  const index = result.handedness.findIndex(
    (categories) => categories[0]?.categoryName.toLowerCase() === 'right',
  )
  return index < 0 ? null : (result.landmarks[index]?.[0]?.y ?? null)
}

export function useRightWristVolume(result: HandLandmarkerResult | null) {
  const smoothedYRef = useRef<number | null>(null)
  const emittedVolumeRef = useRef(0.5)
  const [wristY, setWristY] = useState<number | null>(null)
  const [expressionVolume, setExpressionVolume] = useState(0.5)

  useEffect(() => {
    const rawY = getRightWristY(result)
    if (rawY === null) {
      setWristY(null)
      return
    }

    const clampedY = Math.min(1, Math.max(0, rawY))
    const previousY = smoothedYRef.current
    const nextY =
      previousY === null
        ? clampedY
        : previousY + SMOOTHING_ALPHA * (clampedY - previousY)
    smoothedYRef.current = nextY
    setWristY(nextY)

    const nextVolume = 1 - nextY
    if (
      Math.abs(nextVolume - emittedVolumeRef.current) >= OUTPUT_DEADBAND
    ) {
      emittedVolumeRef.current = nextVolume
      setExpressionVolume(nextVolume)
    }
  }, [result])

  return {
    wristY,
    expressionVolume,
  }
}
