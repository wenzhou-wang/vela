import type { KeyframeTrack } from './KeyframeTrack';

/** A timeline marker on a clip; the mixer fires it when playback crosses `time`. */
export interface AnimationEvent {
  time: number;
  name: string;
}

/** A named set of keyframe tracks with a shared duration. */
export class AnimationClip {
  name: string;
  tracks: KeyframeTrack[];
  duration: number;
  /** Sorted timeline markers (footstep, hit-frame, …) fired by the mixer. */
  readonly events: AnimationEvent[] = [];

  constructor(name: string, tracks: KeyframeTrack[], duration?: number) {
    this.name = name;
    this.tracks = tracks;
    this.duration = duration ?? tracks.reduce((d, t) => Math.max(d, t.duration), 0);
  }

  /** Add a timeline marker at `time` seconds; events stay sorted by time. */
  addEvent(time: number, name: string): this {
    const ev = { time, name };
    let i = this.events.length;
    while (i > 0 && this.events[i - 1].time > time) i--;
    this.events.splice(i, 0, ev);
    return this;
  }
}
