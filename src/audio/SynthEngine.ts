interface VoiceGroup {
  gain: GainNode
  filter: BiquadFilterNode
  oscillators: OscillatorNode[]
}

const ATTACK_SECONDS = 0.08
const RELEASE_SECONDS = 0.18
const SILENCE = 0.0001
const MAX_MASTER_GAIN = 0.22

export class SynthEngine {
  private context: AudioContext | null = null
  private masterGain: GainNode | null = null
  private activeGroup: VoiceGroup | null = null
  private volume = 0.65

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

  playChord(frequencies: readonly number[]) {
    if (
      !this.context ||
      !this.masterGain ||
      this.context.state !== 'running' ||
      frequencies.length === 0
    ) {
      return
    }

    const now = this.context.currentTime
    const previousGroup = this.activeGroup
    const gain = this.context.createGain()
    const filter = this.context.createBiquadFilter()
    const oscillators: OscillatorNode[] = []

    filter.type = 'lowpass'
    filter.frequency.value = 2200
    filter.Q.value = 0.8
    filter.connect(gain)
    gain.connect(this.masterGain)
    gain.gain.setValueAtTime(SILENCE, now)
    gain.gain.exponentialRampToValueAtTime(1, now + ATTACK_SECONDS)

    frequencies.forEach((frequency) => {
      const triangle = this.context!.createOscillator()
      triangle.type = 'triangle'
      triangle.frequency.value = frequency
      triangle.connect(filter)
      triangle.start(now)
      oscillators.push(triangle)

      const saw = this.context!.createOscillator()
      saw.type = 'sawtooth'
      saw.frequency.value = frequency
      saw.detune.value = 7
      saw.connect(filter)
      saw.start(now)
      oscillators.push(saw)
    })

    this.activeGroup = { gain, filter, oscillators }
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
        group.filter.disconnect()
        group.gain.disconnect()
      },
      (releaseSeconds + 0.08) * 1000,
    )
  }
}
