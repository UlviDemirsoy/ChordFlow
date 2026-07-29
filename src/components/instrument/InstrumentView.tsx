import { useEffect, useRef, useState } from 'react'
import { CameraStage } from '../CameraStage'
import { useHandTracking } from '../../hooks/useHandTracking'
import { useLeftHandClassification } from '../../hooks/useLeftHandClassification'
import { useRightHandClassification } from '../../hooks/useRightHandClassification'
import { useGestureInstrument } from '../../hooks/useGestureInstrument'
import { useRightWristVolume } from '../../hooks/useRightWristVolume'
import {
  getChordQualityLabel,
  getScaleDegreeNotes,
  RIGHT_HAND_CHORD_MAP,
  TONIC_OPTIONS,
  type ScaleMode,
  type Tonic,
} from '../../music/chordMap'
import '../../App.css'
import './InstrumentView.css'

function stableLabelText(label: number | null, prefix: string) {
  if (label === null) {
    return 'Bekleniyor'
  }
  if (label === 0) {
    return 'Unknown'
  }
  return `${prefix} ${label}`
}

export function InstrumentView() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const startingRef = useRef(false)
  const [tonic, setTonic] = useState<Tonic>('C')
  const [mode, setMode] = useState<ScaleMode>('major')
  const [audioError, setAudioError] = useState<string | null>(null)
  const {
    status: trackingStatus,
    error: trackingError,
    result,
    cameraActive,
    start,
  } = useHandTracking({ videoRef })
  const left = useLeftHandClassification(result)
  const right = useRightHandClassification(result)
  const wristVolume = useRightWristVolume(result)
  const instrument = useGestureInstrument({
    leftStableLabel: left.stableLabel,
    rightStableLabel: right.stableLabel,
    cameraActive,
    tonic,
    mode,
    expressionVolume: wristVolume.expressionVolume,
  })

  const modelsReady =
    left.modelStatus === 'ready' && right.modelStatus === 'ready'
  const combinedError =
    trackingError || left.error || right.error || audioError || null
  const scaleDegrees = getScaleDegreeNotes(tonic, mode)

  const unlockAudioRef = useRef(instrument.unlockAudio)
  unlockAudioRef.current = instrument.unlockAudio

  useEffect(() => {
    let cancelled = false

    const boot = async () => {
      if (!modelsReady || cancelled) {
        return
      }

      setAudioError(null)
      try {
        await unlockAudioRef.current()
      } catch (error: unknown) {
        if (!cancelled) {
          setAudioError(
            error instanceof Error ? error.message : 'Ses motoru başlatılamadı.',
          )
        }
      }

      if (
        cancelled ||
        cameraActive ||
        startingRef.current ||
        trackingStatus === 'requesting-camera' ||
        trackingStatus === 'running'
      ) {
        return
      }

      startingRef.current = true
      try {
        await start()
      } catch (error: unknown) {
        if (!cancelled) {
          setAudioError(
            error instanceof Error ? error.message : 'Kamera başlatılamadı.',
          )
        }
      } finally {
        startingRef.current = false
      }
    }

    void boot()
    return () => {
      cancelled = true
    }
  }, [cameraActive, modelsReady, start, trackingStatus])

  useEffect(() => {
    if (!cameraActive || instrument.audioReady) {
      return
    }

    const intervalId = window.setInterval(() => {
      void unlockAudioRef.current().catch(() => undefined)
    }, 500)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [cameraActive, instrument.audioReady])

  return (
    <main className="instrument-shell">
      <header className="instrument-header">
        <a className="brand" href="/play" aria-label="ChordFlow ana sayfa">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ChordFlow</strong>
            <small>Gesture Synth</small>
          </span>
        </a>
        <span className="instrument-route">/play</span>
      </header>

      <CameraStage
        videoRef={videoRef}
        result={result}
        status={trackingStatus}
        cameraActive={cameraActive}
      />

      <aside className="instrument-panel">
        <div className="instrument-panel-header">
          <div>
            <p className="eyebrow">TWO-HAND INSTRUMENT</p>
            <h1>Gesture Synth</h1>
          </div>
          <span
            className={`instrument-status instrument-status--${instrument.status}`}
          >
            {instrument.status}
          </span>
        </div>

        <section
          className={`active-chord${instrument.activeChord ? ' is-playing' : ''}`}
        >
          <span>AKTİF AKOR</span>
          <strong>{instrument.activeChord?.displayName ?? '—'}</strong>
          <small>
            {instrument.activeChord
              ? `${instrument.activeChord.rootName} · ${getChordQualityLabel(instrument.activeChord.quality)}`
              : cameraActive
                ? 'İki elden stabil sinyal bekleniyor'
                : modelsReady
                  ? 'Kamera açılıyor...'
                  : 'Modeller yükleniyor...'}
          </small>
        </section>

        <div className="tonality-controls">
          <label>
            <span>Tonal kök</span>
            <select
              value={tonic}
              onChange={(event) => setTonic(event.target.value as Tonic)}
            >
              {TONIC_OPTIONS.map((note) => (
                <option key={note} value={note}>
                  {note}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Gam</span>
            <select
              value={mode}
              onChange={(event) =>
                setMode(event.target.value as ScaleMode)
              }
            >
              <option value="major">Major</option>
              <option value="minor">Minor</option>
            </select>
          </label>
        </div>

        <div className="degree-map">
          <span>Sol el dereceleri · {tonic} {mode}</span>
          <div className="degree-map-grid">
            {scaleDegrees.map(({ degree, note }) => (
              <div
                key={degree}
                className={`degree-box${left.stableLabel === degree ? ' is-active' : ''}`}
              >
                <b>{degree}</b>
                <strong>{note}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="hand-signals">
          <div>
            <span>Sol el · kök derece</span>
            <strong>{stableLabelText(left.stableLabel, 'Derece')}</strong>
            <small>
              {left.prediction
                ? `${(left.prediction.confidence * 100).toFixed(1)}% confidence`
                : 'Sol el yok'}
            </small>
          </div>
          <div>
            <span>Sağ el · akor tipi</span>
            <strong>
              {right.stableLabel
                ? getChordQualityLabel(
                    RIGHT_HAND_CHORD_MAP[right.stableLabel],
                  )
                : stableLabelText(right.stableLabel, 'Class')}
            </strong>
            <small>
              {right.prediction
                ? `${(right.prediction.confidence * 100).toFixed(1)}% confidence`
                : 'Sağ el yok'}
            </small>
          </div>
        </div>

        <label className="volume-control">
          <span>
            Sağ wrist volume
            <b>{Math.round(instrument.expressionVolume * 100)}%</b>
          </span>
          <meter
            min="0"
            max="1"
            value={instrument.expressionVolume}
            aria-label="Sağ wrist expression volume"
          />
          <small>
            {wristVolume.wristY === null
              ? 'Sağ el bekleniyor'
              : `Filtreli wrist Y: ${wristVolume.wristY.toFixed(3)}`}
          </small>
        </label>

        <label className="volume-control">
          <span>
            Master limit
            <b>{Math.round(instrument.volume * 100)}%</b>
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={instrument.volume}
            onChange={(event) =>
              instrument.setVolume(Number(event.target.value))
            }
          />
        </label>

        <div className="gesture-map">
          <span>Sağ el sınıfları</span>
          <p>1 Major · 2 Sus4 · 3 Dominant 7 · 4 Minor · 5 Diminished</p>
        </div>

        {combinedError && (
          <p className="instrument-error" role="alert">
            {combinedError}
          </p>
        )}
      </aside>
    </main>
  )
}
