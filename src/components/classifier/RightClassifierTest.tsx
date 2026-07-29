import { useRef } from 'react'
import { CameraStage } from '../CameraStage'
import { useHandTracking } from '../../hooks/useHandTracking'
import { useRightHandClassification } from '../../hooks/useRightHandClassification'
import { RIGHT_HAND_CLASS_LABELS } from '../../classifier/rightHandClassifier'
import '../../App.css'
import './ClassifierTest.css'

export function RightClassifierTest() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const {
    status,
    error: trackingError,
    result,
    cameraActive,
    start,
    stop,
  } = useHandTracking({ videoRef })
  const {
    modelStatus,
    error: classifierError,
    prediction,
    stableLabel,
    rightHandScore,
  } = useRightHandClassification(result)

  const modelReady = modelStatus === 'ready'
  const acceptedLabel = prediction?.accepted ? prediction.label : null
  const hasAcceptedLabel = acceptedLabel !== null

  return (
    <main className="classifier-shell">
      <header className="classifier-header">
        <a className="brand" href="/" aria-label="ChordFlow ana sayfa">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ChordFlow</strong>
            <small>Realtime right-hand classifier</small>
          </span>
        </a>
        <span className="classifier-route">/classify/right</span>
      </header>

      <CameraStage
        videoRef={videoRef}
        result={result}
        status={status}
        cameraActive={cameraActive}
      />

      <div className="classifier-controls">
        {cameraActive ? (
          <button className="control-button control-button--stop" onClick={stop}>
            <span aria-hidden="true">■</span>
            Kamerayı durdur
          </button>
        ) : (
          <button
            className="control-button"
            onClick={start}
            disabled={!modelReady || status === 'requesting-camera'}
          >
            <span aria-hidden="true">●</span>
            {modelStatus === 'loading'
              ? 'ONNX yükleniyor...'
              : 'Kamerayı başlat'}
          </button>
        )}
      </div>

      <aside className="classifier-panel">
        <div className="classifier-panel-header">
          <div>
            <p className="eyebrow">ONNX Runtime Web</p>
            <h1>Sağ el tahmini</h1>
          </div>
          <span className={`model-status model-status--${modelStatus}`}>
            {modelStatus}
          </span>
        </div>

        <section
          className={`prediction-card${hasAcceptedLabel ? ' is-accepted' : ''}`}
        >
          <span>ANLIK SONUÇ</span>
          <strong>
            {acceptedLabel === 0
              ? 'UNKNOWN'
              : acceptedLabel ?? (prediction ? 'KARARSIZ' : '—')}
          </strong>
          <small>
            {prediction
              ? hasAcceptedLabel
                ? acceptedLabel === 0
                  ? 'Explicit unknown sınıfı'
                  : `Dinamik Class ${acceptedLabel}`
                : `Ham aday: Class ${prediction.label}`
              : 'Sağ el bekleniyor'}
          </small>
        </section>

        <div className="classifier-metrics">
          <div>
            <span>Confidence</span>
            <strong>
              {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '—'}
            </strong>
          </div>
          <div>
            <span>Margin</span>
            <strong>
              {prediction ? `${(prediction.margin * 100).toFixed(1)}%` : '—'}
            </strong>
          </div>
          <div>
            <span>ONNX süre</span>
            <strong>
              {prediction ? `${prediction.inferenceTimeMs.toFixed(1)} ms` : '—'}
            </strong>
          </div>
        </div>

        <div className="stable-result">
          <span>Temporal sonuç · 3/5 çoğunluk</span>
          <strong>
            {stableLabel === 0
              ? 'UNKNOWN'
              : stableLabel !== null
                ? `Class ${stableLabel}`
                : 'Bekleniyor'}
          </strong>
        </div>

        <div className="probability-list">
          <header>
            <span>Sınıf olasılıkları</span>
            <span>
              {rightHandScore
                ? `RIGHT ${(rightHandScore * 100).toFixed(1)}%`
                : 'RIGHT YOK'}
            </span>
          </header>
          {RIGHT_HAND_CLASS_LABELS.map((label, index) => {
            const probability = prediction?.probabilities[index] ?? 0
            return (
              <div className="probability-row" key={label}>
                <b>{label === 0 ? 'U' : label}</b>
                <div>
                  <span style={{ width: `${probability * 100}%` }} />
                </div>
                <code>{(probability * 100).toFixed(1)}%</code>
              </div>
            )
          })}
        </div>

        {(trackingError || classifierError) && (
          <p className="classifier-error" role="alert">
            {trackingError || classifierError}
          </p>
        )}

        <p className="classifier-note">
          Class 0 explicit UNKNOWN sonucudur. Class 1–5 dinamik kimliklerdir;
          akor türü eşlemesi daha sonra değiştirilebilir. Sağ elde confidence
          reddi uygulanmaz; her frame'in argmax sınıfı temporal oylamaya girer.
        </p>
      </aside>
    </main>
  )
}
