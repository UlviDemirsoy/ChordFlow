import { useEffect, useRef, useState } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'
import {
  RightHandClassifier,
  type RightHandClass,
  type RightHandPrediction,
} from '../classifier/rightHandClassifier'

type ModelStatus = 'loading' | 'ready' | 'error'

const INFERENCE_INTERVAL_MS = 100
const HISTORY_SIZE = 5
const REQUIRED_VOTES = 3

function findRightHand(result: HandLandmarkerResult | null) {
  if (!result) {
    return null
  }

  const index = result.handedness.findIndex(
    (categories) => categories[0]?.categoryName.toLowerCase() === 'right',
  )
  if (index < 0) {
    return null
  }

  return {
    landmarks: result.landmarks[index],
    score: result.handedness[index][0]?.score ?? 0,
  }
}

export function useRightHandClassification(
  result: HandLandmarkerResult | null,
) {
  const classifierRef = useRef<RightHandClassifier | null>(null)
  const inFlightRef = useRef(false)
  const lastInferenceAtRef = useRef(0)
  const historyRef = useRef<Array<RightHandClass | null>>([])
  const [modelStatus, setModelStatus] = useState<ModelStatus>('loading')
  const [error, setError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<RightHandPrediction | null>(null)
  const [stableLabel, setStableLabel] = useState<RightHandClass | null>(null)

  if (!classifierRef.current) {
    classifierRef.current = new RightHandClassifier()
  }

  useEffect(() => {
    let active = true
    classifierRef.current
      ?.load()
      .then(() => {
        if (active) {
          setModelStatus('ready')
        }
      })
      .catch((loadError: unknown) => {
        if (active) {
          setModelStatus('error')
          setError(
            loadError instanceof Error
              ? loadError.message
              : 'Sağ el ONNX modeli yüklenemedi.',
          )
        }
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const rightHand = findRightHand(result)
    if (!rightHand) {
      setPrediction(null)
      setStableLabel(null)
      historyRef.current = []
      return
    }

    const now = performance.now()
    if (
      modelStatus !== 'ready' ||
      inFlightRef.current ||
      now - lastInferenceAtRef.current < INFERENCE_INTERVAL_MS
    ) {
      return
    }

    inFlightRef.current = true
    lastInferenceAtRef.current = now
    classifierRef.current
      ?.predict(rightHand.landmarks)
      .then((nextPrediction) => {
        setPrediction(nextPrediction)
        setError(null)

        const history = [
          ...historyRef.current,
          nextPrediction.label,
        ].slice(-HISTORY_SIZE)
        historyRef.current = history

        const voteCounts = new Map<RightHandClass, number>()
        history.forEach((label) => {
          if (label !== null) {
            voteCounts.set(label, (voteCounts.get(label) ?? 0) + 1)
          }
        })
        const winner = [...voteCounts.entries()].sort(
          (left, right) => right[1] - left[1],
        )[0]
        if (history.length === HISTORY_SIZE && winner?.[1] >= REQUIRED_VOTES) {
          setStableLabel((current) =>
            current === winner[0] ? current : winner[0],
          )
        }
      })
      .catch((inferenceError: unknown) => {
        setError(
          inferenceError instanceof Error
            ? inferenceError.message
            : 'Sağ el ONNX inference başarısız.',
        )
      })
      .finally(() => {
        inFlightRef.current = false
      })
  }, [modelStatus, result])

  const rightHand = findRightHand(result)
  return {
    modelStatus,
    error,
    prediction,
    stableLabel,
    rightHandScore: rightHand?.score ?? null,
  }
}
