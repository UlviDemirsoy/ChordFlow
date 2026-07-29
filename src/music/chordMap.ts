import type { LeftHandDegree } from '../classifier/leftHandClassifier'
import type { RightHandClass } from '../classifier/rightHandClassifier'

export const TONIC_OPTIONS = [
  'C',
  'C#',
  'D',
  'D#',
  'E',
  'F',
  'F#',
  'G',
  'G#',
  'A',
  'A#',
  'B',
] as const

export type Tonic = (typeof TONIC_OPTIONS)[number]
export type ScaleMode = 'major' | 'minor'
export type ChordQuality =
  | 'major'
  | 'sus'
  | 'dominant'
  | 'minor'
  | 'diminished'

export interface ResolvedChord {
  degree: Exclude<LeftHandDegree, 0>
  rightClass: Exclude<RightHandClass, 0>
  rootName: string
  quality: ChordQuality
  displayName: string
  midiNotes: number[]
  frequencies: number[]
}

const SCALE_INTERVALS: Record<ScaleMode, readonly number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
}

const CHORD_INTERVALS: Record<ChordQuality, readonly number[]> = {
  major: [0, 4, 7],
  sus: [0, 5, 7],
  dominant: [0, 4, 7, 10],
  minor: [0, 3, 7],
  diminished: [0, 3, 6],
}

export const RIGHT_HAND_CHORD_MAP: Record<
  Exclude<RightHandClass, 0>,
  ChordQuality
> = {
  1: 'major',
  2: 'sus',
  3: 'dominant',
  4: 'minor',
  5: 'diminished',
}

const QUALITY_SUFFIX: Record<ChordQuality, string> = {
  major: '',
  sus: 'sus4',
  dominant: '7',
  minor: 'm',
  diminished: 'dim',
}

const QUALITY_LABEL: Record<ChordQuality, string> = {
  major: 'Major',
  sus: 'Sus4',
  dominant: 'Dominant 7',
  minor: 'Minor',
  diminished: 'Diminished',
}

function midiToFrequency(midi: number) {
  return 440 * 2 ** ((midi - 69) / 12)
}

function noteName(pitchClass: number) {
  return TONIC_OPTIONS[((pitchClass % 12) + 12) % 12]
}

export function getChordQualityLabel(quality: ChordQuality) {
  return QUALITY_LABEL[quality]
}

export function getScaleDegreeNotes(tonic: Tonic, mode: ScaleMode) {
  const tonicPitchClass = TONIC_OPTIONS.indexOf(tonic)
  return SCALE_INTERVALS[mode].map((interval, index) => ({
    degree: (index + 1) as Exclude<LeftHandDegree, 0>,
    note: noteName(tonicPitchClass + interval),
  }))
}

export function resolveGestureChord(
  tonic: Tonic,
  mode: ScaleMode,
  leftDegree: LeftHandDegree | null,
  rightClass: RightHandClass | null,
  octave = 3,
): ResolvedChord | null {
  if (
    leftDegree === null ||
    leftDegree === 0 ||
    rightClass === null ||
    rightClass === 0
  ) {
    return null
  }

  const tonicPitchClass = TONIC_OPTIONS.indexOf(tonic)
  const scaleOffset = SCALE_INTERVALS[mode][leftDegree - 1]
  const absoluteRoot = tonicPitchClass + scaleOffset
  const rootPitchClass = absoluteRoot % 12
  const rootOctaveOffset = Math.floor(absoluteRoot / 12)
  const rootMidi = (octave + 1 + rootOctaveOffset) * 12 + rootPitchClass
  const quality = RIGHT_HAND_CHORD_MAP[rightClass]
  const midiNotes = CHORD_INTERVALS[quality].map(
    (interval) => rootMidi + interval,
  )
  const rootName = noteName(rootPitchClass)

  return {
    degree: leftDegree,
    rightClass,
    rootName,
    quality,
    displayName: `${rootName}${QUALITY_SUFFIX[quality]}`,
    midiNotes,
    frequencies: midiNotes.map(midiToFrequency),
  }
}
