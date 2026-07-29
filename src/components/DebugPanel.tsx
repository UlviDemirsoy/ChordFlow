import { memo } from 'react'
import type { DebugSnapshot } from '../types/handTracking'

interface DebugPanelProps {
  snapshot: DebugSnapshot | null
}

const LANDMARK_NAMES = [
  'Wrist',
  'Thumb CMC',
  'Thumb MCP',
  'Thumb IP',
  'Thumb tip',
  'Index MCP',
  'Index PIP',
  'Index DIP',
  'Index tip',
  'Middle MCP',
  'Middle PIP',
  'Middle DIP',
  'Middle tip',
  'Ring MCP',
  'Ring PIP',
  'Ring DIP',
  'Ring tip',
  'Pinky MCP',
  'Pinky PIP',
  'Pinky DIP',
  'Pinky tip',
]

function formatCoordinate(value: number) {
  return value.toFixed(3)
}

export const DebugPanel = memo(function DebugPanel({
  snapshot,
}: DebugPanelProps) {
  const result = snapshot?.result
  const handCount = result?.landmarks.length ?? 0

  return (
    <aside className="debug-panel" aria-live="polite">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Realtime output</p>
          <h2>Landmark verisi</h2>
        </div>
        <span className="output-rate">10 Hz UI</span>
      </div>

      <div className="metric-grid">
        <div className="metric">
          <span>El sayısı</span>
          <strong>{handCount}</strong>
        </div>
        <div className="metric">
          <span>Inference</span>
          <strong>
            {snapshot ? `${snapshot.inferenceTimeMs.toFixed(1)} ms` : '—'}
          </strong>
        </div>
        <div className="metric">
          <span>Tracking FPS</span>
          <strong>{snapshot ? snapshot.fps.toFixed(1) : '—'}</strong>
        </div>
      </div>

      {handCount === 0 ? (
        <div className="empty-output">
          <div className="scan-line" />
          <p>Henüz el algılanmadı.</p>
          <span>Elini kameranın görüş alanında açık biçimde tut.</span>
        </div>
      ) : (
        <div className="hands-output">
          {result?.landmarks.map((landmarks, handIndex) => {
            const handedness = result.handedness[handIndex]?.[0]

            return (
              <section className="hand-output" key={`hand-${handIndex}`}>
                <header>
                  <div>
                    <span>HAND {handIndex + 1}</span>
                    <h3>{handedness?.categoryName || 'Unknown'}</h3>
                  </div>
                  <strong>
                    {handedness
                      ? `${(handedness.score * 100).toFixed(1)}%`
                      : '—'}
                  </strong>
                </header>

                <div className="coordinate-header" aria-hidden="true">
                  <span>Landmark</span>
                  <span>X</span>
                  <span>Y</span>
                  <span>Z</span>
                </div>
                <ol className="coordinate-list">
                  {landmarks.map((landmark, landmarkIndex) => (
                    <li key={LANDMARK_NAMES[landmarkIndex]}>
                      <span>
                        <b>{landmarkIndex.toString().padStart(2, '0')}</b>
                        {LANDMARK_NAMES[landmarkIndex]}
                      </span>
                      <code>{formatCoordinate(landmark.x)}</code>
                      <code>{formatCoordinate(landmark.y)}</code>
                      <code>{formatCoordinate(landmark.z)}</code>
                    </li>
                  ))}
                </ol>
              </section>
            )
          })}
        </div>
      )}
    </aside>
  )
})
