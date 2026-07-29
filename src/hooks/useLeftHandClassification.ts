import { useEffect, useRef, useState } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'
import {
  LeftHandClassifier,
  type LeftHandDegree,
  type LeftHandPrediction,
} from '../classifier/leftHandClassifier'

type ModelStatus = 'loading' | 'ready' | 'error'

const INFERENCE_INTERVAL_MS = 100
const HISTORY_SIZE = 5
const REQUIRED_VOTES = 3

function findLeftHand(result: HandLandmarkerResult | null) {
  if (!result) {
    return null
  }

  const index = result.handedness.findIndex(
    (categories) => categories[0]?.categoryName.toLowerCase() === 'left',
  )
  if (index < 0) {
    return null
  }

  return {
    landmarks: result.landmarks[index],
    score: result.handedness[index][0]?.score ?? 0,
  }
}

export function useLeftHandClassification(
  result: HandLandmarkerResult | null,
) {
  const classifierRef = useRef<LeftHandClassifier | null>(null)
  const inFlightRef = useRef(false)
  const lastInferenceAtRef = useRef(0)
  const historyRef = useRef<Array<LeftHandDegree | null>>([])
  const [modelStatus, setModelStatus] = useState<ModelStatus>('loading')
  const [error, setError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<LeftHandPrediction | null>(null)
  const [stableLabel, setStableLabel] = useState<LeftHandDegree | null>(null)

  if (!classifierRef.current) {
    classifierRef.current = new LeftHandClassifier()
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
              : 'ONNX modeli yüklenemedi.',
          )
        }
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const leftHand = findLeftHand(result)
    if (!leftHand) {
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
      ?.predict(leftHand.landmarks)
      .then((nextPrediction) => {
        setPrediction(nextPrediction)
        setError(null)

        const history = [
          ...historyRef.current,
          nextPrediction.label === 0
            ? 0
            : nextPrediction.accepted
              ? nextPrediction.label
              : null,
        ].slice(-HISTORY_SIZE)
        historyRef.current = history

        const voteCounts = new Map<LeftHandDegree, number>()
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
            : 'ONNX inference başarısız.',
        )
      })
      .finally(() => {
        inFlightRef.current = false
      })
  }, [modelStatus, result])

  const leftHand = findLeftHand(result)

  return {
    modelStatus,
    error,
    prediction,
    stableLabel,
    leftHandScore: leftHand?.score ?? null,
  }
}
