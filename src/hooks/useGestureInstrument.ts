import { useCallback, useEffect, useRef, useState } from 'react'
import type { LeftHandDegree } from '../classifier/leftHandClassifier'
import type { RightHandClass } from '../classifier/rightHandClassifier'
import { SynthEngine } from '../audio/SynthEngine'
import {
  resolveGestureChord,
  type ResolvedChord,
  type ScaleMode,
  type Tonic,
} from '../music/chordMap'

export type InstrumentStatus = 'idle' | 'armed' | 'playing' | 'muted'

interface UseGestureInstrumentInput {
  leftStableLabel: LeftHandDegree | null
  rightStableLabel: RightHandClass | null
  cameraActive: boolean
  tonic: Tonic
  mode: ScaleMode
  expressionVolume: number
  /** Right palm tilt 0–1 → oscillator mix */
  handTilt: number
}

export function useGestureInstrument({
  leftStableLabel,
  rightStableLabel,
  cameraActive,
  tonic,
  mode,
  expressionVolume,
  handTilt,
}: UseGestureInstrumentInput) {
  const engineRef = useRef<SynthEngine | null>(null)
  const chordKeyRef = useRef<string | null>(null)
  const [audioReady, setAudioReady] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolumeState] = useState(0.65)
  const [activeChord, setActiveChord] = useState<ResolvedChord | null>(null)
  const [pageVisible, setPageVisible] = useState(!document.hidden)

  if (!engineRef.current) {
    engineRef.current = new SynthEngine()
  }

  const unlockAudio = useCallback(async () => {
    await engineRef.current?.unlock()
    engineRef.current?.setVolume(volume * expressionVolume)
    setAudioReady(true)
  }, [expressionVolume, volume])

  const stopAudio = useCallback(async () => {
    chordKeyRef.current = null
    setActiveChord(null)
    setAudioReady(false)
    await engineRef.current?.suspend()
  }, [])

  const toggleMuted = useCallback(() => {
    setMuted((current) => !current)
  }, [])

  const setVolume = useCallback((nextVolume: number) => {
    setVolumeState(nextVolume)
  }, [])

  useEffect(() => {
    engineRef.current?.setVolume(volume * expressionVolume)
  }, [expressionVolume, volume])

  useEffect(() => {
    engineRef.current?.setTilt(handTilt)
  }, [handTilt])

  useEffect(() => {
    const handleVisibilityChange = () => {
      const visible = !document.hidden
      setPageVisible(visible)
      if (!visible) {
        chordKeyRef.current = null
        setActiveChord(null)
        void engineRef.current?.suspend()
      } else if (audioReady) {
        void engineRef.current?.resume()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [audioReady])

  useEffect(() => {
    const chord = resolveGestureChord(
      tonic,
      mode,
      leftStableLabel,
      rightStableLabel,
    )

    if (
      !audioReady ||
      !cameraActive ||
      !pageVisible ||
      muted ||
      !chord
    ) {
      if (chordKeyRef.current !== null) {
        engineRef.current?.stopAll()
      }
      chordKeyRef.current = null
      setActiveChord(null)
      return
    }

    const chordKey = chord.midiNotes.join('-')
    if (chordKeyRef.current === chordKey) {
      return
    }

    let cancelled = false
    void engineRef.current?.resume().then(() => {
      if (!cancelled) {
        engineRef.current?.playChord(chord.frequencies)
        chordKeyRef.current = chordKey
        setActiveChord(chord)
      }
    })

    return () => {
      cancelled = true
    }
  }, [
    audioReady,
    cameraActive,
    leftStableLabel,
    mode,
    muted,
    pageVisible,
    rightStableLabel,
    tonic,
  ])

  useEffect(
    () => () => {
      engineRef.current?.dispose()
    },
    [],
  )

  const status: InstrumentStatus = !cameraActive
    ? 'idle'
    : muted
      ? 'muted'
      : activeChord
        ? 'playing'
        : 'armed'

  return {
    status,
    audioReady,
    muted,
    volume,
    expressionVolume,
    handTilt,
    activeChord,
    unlockAudio,
    stopAudio,
    toggleMuted,
    setVolume,
  }
}
