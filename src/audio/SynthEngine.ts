interface VoiceGroup {
  gain: GainNode
  filter: BiquadFilterNode
  triangleGain: GainNode
  sawGain: GainNode
  squareGain: GainNode
  oscillators: OscillatorNode[]
}

const ATTACK_SECONDS = 0.08
const RELEASE_SECONDS = 0.18
const SILENCE = 0.0001
const MAX_MASTER_GAIN = 0.22

function mixFromTilt(tilt: number) {
  const t = Math.min(1, Math.max(0, tilt))
  return {
    triangle: 0.9 - 0.45 * t,
    saw: 0.2 + 0.7 * t,
    square: Math.max(0, (t - 0.25) / 0.75) * 0.55,
  }
}

export class SynthEngine {
  private context: AudioContext | null = null
  private masterGain: GainNode | null = null
  private activeGroup: VoiceGroup | null = null
  private volume = 0.65
  private tilt = 0

  get state() {
    return this.context?.state ?? 'closed'
  }

  async unlock() {
    if (!this.context || this.context.state === 'closed') {
      this.context = new AudioContext({ latencyHint: 'interactive' })
      this.masterGain = this.context.createGain()
      this.masterGain.gain.value = this.volume * MAX_MASTER_GAIN
      this.masterGain.connect(this.context.destination)
    }

    if (this.context.state === 'suspended') {
      await this.context.resume()
    }
  }

  async resume() {
    if (this.context?.state === 'suspended') {
      await this.context.resume()
    }
  }

  setVolume(volume: number) {
    this.volume = Math.min(1, Math.max(0, volume))
    if (!this.context || !this.masterGain) {
      return
    }

    const now = this.context.currentTime
    this.masterGain.gain.cancelScheduledValues(now)
    this.masterGain.gain.setTargetAtTime(
      this.volume * MAX_MASTER_GAIN,
      now,
      0.025,
    )
  }

  /** Right-hand palm tilt 0–1 → triangle / saw / square mix. */
  setTilt(tilt: number) {
    this.tilt = Math.min(1, Math.max(0, tilt))
    if (!this.context || !this.activeGroup) {
      return
    }

    const mix = mixFromTilt(this.tilt)
    const now = this.context.currentTime
    const group = this.activeGroup
    group.triangleGain.gain.setTargetAtTime(mix.triangle, now, 0.04)
    group.sawGain.gain.setTargetAtTime(mix.saw, now, 0.04)
    group.squareGain.gain.setTargetAtTime(mix.square, now, 0.04)
    group.filter.frequency.setTargetAtTime(1800 + this.tilt * 1400, now, 0.05)
    group.filter.Q.setTargetAtTime(0.7 + this.tilt * 0.5, now, 0.05)
  }

  playChord(frequencies: readonly number[]) {
    if (!this.context || !this.masterGain || frequencies.length === 0) {
      return
    }

    if (this.context.state === 'suspended') {
      void this.context.resume()
    }

    if (this.context.state !== 'running' && this.context.state !== 'suspended') {
      return
    }

    const now = this.context.currentTime
    const previousGroup = this.activeGroup
    const mix = mixFromTilt(this.tilt)
    const gain = this.context.createGain()
    const filter = this.context.createBiquadFilter()
    const triangleGain = this.context.createGain()
    const sawGain = this.context.createGain()
    const squareGain = this.context.createGain()
    const oscillators: OscillatorNode[] = []

    filter.type = 'lowpass'
    filter.frequency.value = 1800 + this.tilt * 1400
    filter.Q.value = 0.7 + this.tilt * 0.5
    filter.connect(gain)
    gain.connect(this.masterGain)
    gain.gain.setValueAtTime(SILENCE, now)
    gain.gain.exponentialRampToValueAtTime(1, now + ATTACK_SECONDS)

    triangleGain.gain.value = mix.triangle
    sawGain.gain.value = mix.saw
    squareGain.gain.value = mix.square
    triangleGain.connect(filter)
    sawGain.connect(filter)
    squareGain.connect(filter)

    frequencies.forEach((frequency) => {
      const triangle = this.context!.createOscillator()
      triangle.type = 'triangle'
      triangle.frequency.value = frequency
      triangle.connect(triangleGain)
      triangle.start(now)
      oscillators.push(triangle)

      const saw = this.context!.createOscillator()
      saw.type = 'sawtooth'
      saw.frequency.value = frequency
      saw.detune.value = 7
      saw.connect(sawGain)
      saw.start(now)
      oscillators.push(saw)

      const square = this.context!.createOscillator()
      square.type = 'square'
      square.frequency.value = frequency
      square.detune.value = -5
      square.connect(squareGain)
      square.start(now)
      oscillators.push(square)
    })

    this.activeGroup = {
      gain,
      filter,
      triangleGain,
      sawGain,
      squareGain,
      oscillators,
    }
    if (previousGroup) {
      this.releaseGroup(previousGroup, now)
    }
  }

  stopAll(releaseSeconds = RELEASE_SECONDS) {
    if (!this.context || !this.activeGroup) {
      return
    }

    const group = this.activeGroup
    this.activeGroup = null
    this.releaseGroup(group, this.context.currentTime, releaseSeconds)
  }

  async suspend() {
    this.stopAll(0.08)
    if (this.context?.state === 'running') {
      await this.context.suspend()
    }
  }

  dispose() {
    this.stopAll(0.03)
    this.masterGain?.disconnect()
    this.masterGain = null
    const context = this.context
    this.context = null
    if (context && context.state !== 'closed') {
      void context.close()
    }
  }

  private releaseGroup(
    group: VoiceGroup,
    now: number,
    releaseSeconds = RELEASE_SECONDS,
  ) {
    group.gain.gain.cancelScheduledValues(now)
    group.gain.gain.setValueAtTime(
      Math.max(SILENCE, group.gain.gain.value),
      now,
    )
    group.gain.gain.exponentialRampToValueAtTime(
      SILENCE,
      now + releaseSeconds,
    )
    group.oscillators.forEach((oscillator) => {
      oscillator.stop(now + releaseSeconds + 0.03)
      oscillator.addEventListener(
        'ended',
        () => {
          oscillator.disconnect()
        },
        { once: true },
      )
    })
    window.setTimeout(
      () => {
        group.triangleGain.disconnect()
        group.sawGain.disconnect()
        group.squareGain.disconnect()
        group.filter.disconnect()
        group.gain.disconnect()
      },
      (releaseSeconds + 0.08) * 1000,
    )
  }
}
