import { useCallback, useEffect, useRef, useState } from 'react'
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision'
import { CameraStage } from '../CameraStage'
import { useHandTracking } from '../../hooks/useHandTracking'
import {
  captureVideoAsJpeg,
  getDatasetCounts,
  saveHandSample,
  type DatasetCounts,
  type DatasetHand,
  type DegreeLabel,
} from '../../dataset/datasetApi'
import '../../App.css'
import './DatasetCollector.css'

const LEFT_CLASS_LABELS: DegreeLabel[] = [0, 1, 2, 3, 4, 5, 6, 7]
const RIGHT_CLASS_LABELS: DegreeLabel[] = [0, 1, 2, 3, 4, 5]

interface HandDatasetCollectorProps {
  hand: DatasetHand
  classLabels: DegreeLabel[]
}

function formatLabel(label: DegreeLabel, hand: DatasetHand) {
  if (label === 0) {
    return 'Unknown'
  }
  return hand === 'left' ? `${label}. derece` : `Class ${label}`
}

function getHand(result: HandLandmarkerResult | null, hand: DatasetHand) {
  if (!result) {
    return null
  }

  const index = result.handedness.findIndex(
    (categories) => categories[0]?.categoryName.toLowerCase() === hand,
  )

  if (index < 0) {
    return null
  }

  return {
    landmarks: result.landmarks[index],
    score: result.handedness[index][0]?.score ?? 0,
  }
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function HandDatasetCollector({
  hand,
  classLabels,
}: HandDatasetCollectorProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const latestResultRef = useRef<HandLandmarkerResult | null>(null)
  const cancelBurstRef = useRef(false)
  const savingLockRef = useRef(false)
  const [selectedDegree, setSelectedDegree] = useState<DegreeLabel>(0)
  const [burstSize, setBurstSize] = useState(25)
  const [counts, setCounts] = useState<DatasetCounts>({
    counts: Object.fromEntries(classLabels.map((label) => [label, 0])),
    total: 0,
  })
  const [message, setMessage] = useState('Kamera başlatılmayı bekliyor.')
  const [isSaving, setIsSaving] = useState(false)
  const [isBursting, setIsBursting] = useState(false)
  const [burstProgress, setBurstProgress] = useState({ current: 0, total: 0 })

  const {
    status,
    error,
    result,
    cameraActive,
    start,
    stop,
  } = useHandTracking({ videoRef })

  latestResultRef.current = result
  const detectedHand = getHand(result, hand)

  const refreshCounts = useCallback(async () => {
    try {
      setCounts(await getDatasetCounts(hand))
    } catch (countError) {
      setMessage(
        countError instanceof Error
          ? countError.message
          : 'Dataset sayaçları okunamadı.',
      )
    }
  }, [hand])

  useEffect(() => {
    void refreshCounts()
  }, [refreshCounts])

  const captureSample = useCallback(async (degree: DegreeLabel) => {
    const video = videoRef.current
    const currentHand = getHand(latestResultRef.current, hand)

    if (!video || !currentHand) {
      throw new Error(
        `${hand === 'left' ? 'Sol' : 'Sağ'} el algılanmıyor; eli tamamen kadraja getir.`,
      )
    }

    if (currentHand.score < 0.7) {
      throw new Error(
        `${hand === 'left' ? 'Sol' : 'Sağ'} el confidence değeri %70 altında.`,
      )
    }

    const imageDataUrl = captureVideoAsJpeg(video)
    const saved = await saveHandSample(hand, {
      degree,
      imageDataUrl,
      landmarks: currentHand.landmarks,
      handednessScore: currentHand.score,
    })

    setCounts((current) => ({
      counts: {
        ...current.counts,
        [degree]: (current.counts[degree] ?? 0) + 1,
      },
      total: current.total + 1,
    }))

    return saved
  }, [hand])

  const captureSingle = useCallback(async () => {
    if (savingLockRef.current || isBursting) {
      return
    }

    savingLockRef.current = true
    setIsSaving(true)
    setMessage(
      `${formatLabel(selectedDegree, hand)} örneği kaydediliyor...`,
    )

    try {
      const saved = await captureSample(selectedDegree)
      setMessage(`Kaydedildi: ${saved.id}`)
    } catch (captureError) {
      setMessage(
        captureError instanceof Error
          ? captureError.message
          : 'Örnek kaydedilemedi.',
      )
    } finally {
      savingLockRef.current = false
      setIsSaving(false)
    }
  }, [captureSample, hand, isBursting, selectedDegree])

  const startBurst = useCallback(async () => {
    if (isSaving || isBursting) {
      return
    }

    const degree = selectedDegree
    cancelBurstRef.current = false
    setIsBursting(true)
    setBurstProgress({ current: 0, total: burstSize })

    let savedCount = 0
    let attempts = 0
    const maxAttempts = burstSize * 5

    while (
      savedCount < burstSize &&
      attempts < maxAttempts &&
      !cancelBurstRef.current
    ) {
      attempts += 1

      try {
        await captureSample(degree)
        savedCount += 1
        setBurstProgress({ current: savedCount, total: burstSize })
        setMessage(
          `${formatLabel(degree, hand)}: ${savedCount}/${burstSize} örnek kaydedildi.`,
        )
      } catch (captureError) {
        setMessage(
          captureError instanceof Error
            ? captureError.message
            : 'Kare atlandı.',
        )
      }

      if (savedCount < burstSize && !cancelBurstRef.current) {
        await wait(350)
      }
    }

    if (cancelBurstRef.current) {
      setMessage(`Toplu kayıt durduruldu. ${savedCount} örnek kaydedildi.`)
    } else if (savedCount === burstSize) {
      setMessage(
        `${formatLabel(degree, hand)} için ${savedCount} örnek tamamlandı.`,
      )
    } else {
      setMessage(
        `${hand === 'left' ? 'Sol' : 'Sağ'} el yeterince algılanamadı. ${savedCount}/${burstSize} kaydedildi.`,
      )
    }

    setIsBursting(false)
  }, [burstSize, captureSample, hand, isBursting, isSaving, selectedDegree])

  const cancelBurst = useCallback(() => {
    cancelBurstRef.current = true
  }, [])

  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement

      if (event.code === 'Space') {
        // Button odaktaysa tarayıcının native click davranışı tek kaydı yapar.
        if (target.tagName === 'BUTTON') {
          return
        }
        event.preventDefault()
        void captureSingle()
        return
      }

      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) {
        return
      }

      const numericDegree = Number(event.key)
      if (classLabels.includes(numericDegree as DegreeLabel)) {
        setSelectedDegree(numericDegree as DegreeLabel)
        setMessage(
          `${formatLabel(numericDegree as DegreeLabel, hand)} seçildi.`,
        )
        return
      }

    }

    window.addEventListener('keydown', handleKeyboard)
    return () => window.removeEventListener('keydown', handleKeyboard)
  }, [captureSingle, classLabels, hand])

  return (
    <main className="collector-shell">
      <header className="collector-header">
        <a className="brand" href="/" aria-label="ChordFlow ana sayfa">
          <span className="brand-mark" aria-hidden="true">
            CF
          </span>
          <span>
            <strong>ChordFlow</strong>
            <small>
              {hand === 'left' ? 'Left-hand' : 'Right-hand'} dataset collector
            </small>
          </span>
        </a>
        <span className="collector-route">
          {hand === 'left' ? '/collect' : '/collect/right'}
        </span>
      </header>

      <CameraStage
        videoRef={videoRef}
        result={result}
        status={status}
        cameraActive={cameraActive}
      />

      <aside className="collector-panel">
        <div className="collector-title">
          <div>
            <p className="eyebrow">Dataset session</p>
            <h1>{hand === 'left' ? 'Sol' : 'Sağ'} el örnekleri</h1>
          </div>
          <span className={detectedHand ? 'hand-ready' : 'hand-missing'}>
            {detectedHand
              ? `${hand.toUpperCase()} ${(detectedHand.score * 100).toFixed(1)}%`
              : `${hand.toUpperCase()} YOK`}
          </span>
        </div>

        <div className="collector-field">
          <label htmlFor="degree">Dataset sınıfı</label>
          <select
            id="degree"
            value={selectedDegree}
            onChange={(event) =>
              setSelectedDegree(Number(event.target.value) as DegreeLabel)
            }
            disabled={isBursting}
          >
            {classLabels.map((label) => (
              <option value={label} key={label}>
                {formatLabel(label, hand)}
              </option>
            ))}
          </select>
          <small>
            Klavyeden 0–{classLabels.at(-1)} ile de seçebilirsin. 0 = Unknown.
          </small>
        </div>

        <div className="collector-actions">
          {!cameraActive ? (
            <button
              className="collector-button collector-button--primary"
              onClick={start}
              disabled={status === 'loading-model'}
            >
              {status === 'loading-model'
                ? 'Model yükleniyor...'
                : 'Kamerayı başlat'}
            </button>
          ) : (
            <>
              <button
                className="collector-button collector-button--primary"
                onClick={captureSingle}
                disabled={!detectedHand || isSaving || isBursting}
              >
                {isSaving ? 'Kaydediliyor...' : 'Tek örnek kaydet'}
                <kbd>Space</kbd>
              </button>

              <div className="burst-controls">
                <select
                  aria-label="Toplu örnek sayısı"
                  value={burstSize}
                  onChange={(event) => setBurstSize(Number(event.target.value))}
                  disabled={isBursting}
                >
                  <option value={10}>10 kare</option>
                  <option value={25}>25 kare</option>
                  <option value={50}>50 kare</option>
                </select>
                {isBursting ? (
                  <button
                    className="collector-button collector-button--danger"
                    onClick={cancelBurst}
                  >
                    Durdur {burstProgress.current}/{burstProgress.total}
                  </button>
                ) : (
                  <button
                    className="collector-button"
                    onClick={startBurst}
                    disabled={!detectedHand}
                  >
                    Toplu kayıt
                  </button>
                )}
              </div>

              <button className="camera-off-button" onClick={stop}>
                Kamerayı kapat
              </button>
            </>
          )}
        </div>

        <p className={`collector-message${error ? ' is-error' : ''}`}>
          {error || message}
        </p>

        <div
          className={`dataset-counts${hand === 'right' ? ' is-right' : ''}`}
        >
          <header>
            <span>Sınıf dağılımı</span>
            <strong>{counts.total} toplam</strong>
          </header>
          <div>
            {classLabels.map((degree) => (
              <span
                className={degree === selectedDegree ? 'is-selected' : ''}
                key={degree}
              >
                <b>{degree === 0 ? 'U' : degree}</b>
                {counts.counts[degree] ?? 0}
              </span>
            ))}
          </div>
        </div>

        <p className="storage-path">
          model/data/{hand}/{selectedDegree}/
        </p>
      </aside>
    </main>
  )
}

export function DatasetCollector() {
  return <HandDatasetCollector hand="left" classLabels={LEFT_CLASS_LABELS} />
}

export function RightDatasetCollector() {
  return <HandDatasetCollector hand="right" classLabels={RIGHT_CLASS_LABELS} />
}
