import { useRef } from 'react'
import { CameraStage } from './components/CameraStage'
import { DebugPanel } from './components/DebugPanel'
import { useHandTracking } from './hooks/useHandTracking'
import './App.css'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const {
    status,
    error,
    result,
    debugSnapshot,
    isRunning,
    cameraActive,
    start,
    stop,
  } = useHandTracking({ videoRef })

  const isBusy =
    status === 'loading-model' || status === 'requesting-camera'

  return (
    <main className="app-shell">
      <header className="app-header">
        <a className="brand" href="/" aria-label="ChordFlow ana sayfa">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ChordFlow</strong>
            <small>Gesture instrument lab</small>
          </span>
        </a>

        <div className="privacy-note">
          <span aria-hidden="true">●</span>
          On-device processing
        </div>
      </header>

      <div className="controls">
        {cameraActive ? (
          <button className="control-button control-button--stop" onClick={stop}>
            <span aria-hidden="true">■</span>
            Kamerayı durdur
          </button>
        ) : (
          <button
            className="control-button"
            onClick={start}
            disabled={isBusy}
          >
            <span aria-hidden="true">●</span>
            {status === 'loading-model'
              ? 'Model yükleniyor...'
              : status === 'requesting-camera'
                ? 'İzin bekleniyor...'
                : 'Kamerayı başlat'}
          </button>
        )}
        <p>{isRunning ? 'Takip aktif' : 'Kamera kapalı'}</p>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <strong>Kamera veya model başlatılamadı.</strong>
          <span>{error}</span>
        </div>
      )}

      <div className="workspace-grid">
        <CameraStage
          videoRef={videoRef}
          result={result}
          status={status}
          cameraActive={cameraActive}
        />
        <DebugPanel snapshot={debugSnapshot} />
      </div>
    </main>
  )
}

export default App
