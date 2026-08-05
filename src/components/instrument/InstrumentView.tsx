import { useEffect, useRef, useState } from 'react'
import { CameraStage } from '../CameraStage'
import { useHandTracking } from '../../hooks/useHandTracking'
import { useLeftHandClassification } from '../../hooks/useLeftHandClassification'
import { useRightHandClassification } from '../../hooks/useRightHandClassification'
import { useGestureInstrument } from '../../hooks/useGestureInstrument'
import { useRightWristVolume } from '../../hooks/useRightWristVolume'
import { useRightHandTilt } from '../../hooks/useRightHandTilt'
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
    return 'Waiting'
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
  const handTilt = useRightHandTilt(result)
  const instrument = useGestureInstrument({
    leftStableLabel: left.stableLabel,
    rightStableLabel: right.stableLabel,
    cameraActive,
    tonic,
    mode,
    expressionVolume: wristVolume.expressionVolume,
    handTilt: handTilt.tilt,
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
            error instanceof Error ? error.message : 'Audio failed to start.',
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
            error instanceof Error ? error.message : 'Camera failed to start.',
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
        <a className="brand" href="/play" aria-label="ChordFlow home">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ChordFlow</strong>
            <small>Gesture Synth</small>
          </span>
        </a>
      </header>

      <CameraStage
        videoRef={videoRef}
        result={result}
        status={trackingStatus}
        cameraActive={cameraActive}
      />

      <aside className="instrument-panel">
        <div className="instrument-panel-header">
          <span className="instrument-panel-title">Now playing</span>
          <span
            className={`instrument-status instrument-status--${instrument.status}`}
          >
            {instrument.status}
          </span>
        </div>

        <section
          className={`active-chord${instrument.activeChord ? ' is-playing' : ''}`}
        >
          <strong>{instrument.activeChord?.displayName ?? '—'}</strong>
          <small>
            {instrument.activeChord
              ? getChordQualityLabel(instrument.activeChord.quality)
              : cameraActive
                ? 'Hold a left and right gesture'
                : modelsReady
                  ? 'Opening camera…'
                  : 'Loading models…'}
          </small>
        </section>

        <div className="tonality-controls">
          <label>
            <span>Key</span>
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
            <span>Mode</span>
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
          <span>Left · degrees</span>
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
            <span>Left</span>
            <strong>{stableLabelText(left.stableLabel, 'Deg')}</strong>
          </div>
          <div>
            <span>Right</span>
            <strong>
              {right.stableLabel
                ? getChordQualityLabel(
                    RIGHT_HAND_CHORD_MAP[right.stableLabel],
                  )
                : stableLabelText(right.stableLabel, 'Class')}
            </strong>
          </div>
        </div>

        <label className="volume-control">
          <span>
            Volume
            <b>{Math.round(instrument.expressionVolume * 100)}%</b>
          </span>
          <meter
            min="0"
            max="1"
            value={instrument.expressionVolume}
            aria-label="Right wrist volume"
          />
        </label>

        <label className="volume-control">
          <span>
            Timbre
            <b>{Math.round(instrument.handTilt * 100)}%</b>
          </span>
          <meter
            min="0"
            max="1"
            value={instrument.handTilt}
            aria-label="Right hand tilt timbre"
          />
        </label>

        {combinedError && (
          <p className="instrument-error" role="alert">
            {combinedError}
          </p>
        )}
      </aside>
    </main>
  )
}
