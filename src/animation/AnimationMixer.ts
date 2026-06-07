import type { AnimationClip } from './AnimationClip';

/**
 * Plays one `AnimationClip` at a time, sampling its tracks into the target
 * nodes each `update`. Lightweight by design — enough to drive glTF transform
 * and skeletal animation in the viewer.
 */
export class AnimationMixer {
  clip: AnimationClip | null = null;
  time = 0;
  timeScale = 1;
  loop = true;
  paused = false;

  play(clip: AnimationClip): this {
    this.clip = clip;
    this.time = 0;
    this.paused = false;
    return this;
  }

  stop(): this {
    this.clip = null;
    this.time = 0;
    return this;
  }

  /** Jump to an absolute time (seconds) and sample. */
  setTime(time: number): this {
    this.time = time;
    this.sample();
    return this;
  }

  /** Advance by `dt` seconds and sample all tracks. */
  update(dt: number): void {
    const clip = this.clip;
    if (!clip || this.paused) return;

    this.time += dt * this.timeScale;
    if (clip.duration > 0) {
      if (this.loop) {
        this.time %= clip.duration;
        if (this.time < 0) this.time += clip.duration;
      } else {
        this.time = Math.max(0, Math.min(this.time, clip.duration));
      }
    }
    this.sample();
  }

  private sample(): void {
    const clip = this.clip;
    if (!clip) return;
    const tracks = clip.tracks;
    for (let i = 0; i < tracks.length; i++) tracks[i].sample(this.time);
  }
}
