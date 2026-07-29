import * as ort from 'onnxruntime-web/wasm'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

export const RIGHT_HAND_CLASS_LABELS = [0, 1, 2, 3, 4, 5] as const
export type RightHandClass = (typeof RIGHT_HAND_CLASS_LABELS)[number]

export interface RightHandPrediction {
  label: RightHandClass
  confidence: number
  margin: number
  probabilities: number[]
  accepted: boolean
  inferenceTimeMs: number
}

const MODEL_URL = '/models/right_hand_model.onnx?v=right-class-blocks-205333'

function softmax(logits: readonly number[]) {
  const maximum = Math.max(...logits)
  const exponentials = logits.map((value) => Math.exp(value - maximum))
  const denominator = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials.map((value) => value / denominator)
}

export class RightHandClassifier {
  private sessionPromise: Promise<ort.InferenceSession> | null = null

  load() {
    if (!this.sessionPromise) {
      this.sessionPromise = ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
    }
    return this.sessionPromise
  }

  async predict(
    landmarks: NormalizedLandmark[],
  ): Promise<RightHandPrediction> {
    if (landmarks.length !== 21) {
      throw new Error('Classifier 21 landmark bekliyor.')
    }

    const session = await this.load()
    const rawFeatures = new Float32Array(63)
    landmarks.forEach((landmark, index) => {
      const offset = index * 3
      rawFeatures[offset] = landmark.x
      rawFeatures[offset + 1] = landmark.y
      rawFeatures[offset + 2] = landmark.z
    })

    const startedAt = performance.now()
    const output = await session.run({
      landmarks: new ort.Tensor('float32', rawFeatures, [1, 63]),
    })
    const logitsTensor = output.logits
    if (!logitsTensor) {
      throw new Error('ONNX modeli logits çıktısı üretmedi.')
    }

    const probabilities = softmax(Array.from(logitsTensor.data as Float32Array))
    const sorted = probabilities
      .map((probability, index) => ({ probability, index }))
      .sort((left, right) => right.probability - left.probability)
    const first = sorted[0]
    const second = sorted[1]
    const margin = first.probability - second.probability

    return {
      label: RIGHT_HAND_CLASS_LABELS[first.index],
      confidence: first.probability,
      margin,
      probabilities,
      accepted: true,
      inferenceTimeMs: performance.now() - startedAt,
    }
  }
}
