import type { KeyframeTrack } from './KeyframeTrack';

/** A named set of keyframe tracks with a shared duration. */
export class AnimationClip {
  name: string;
  tracks: KeyframeTrack[];
  duration: number;

  constructor(name: string, tracks: KeyframeTrack[], duration?: number) {
    this.name = name;
    this.tracks = tracks;
    this.duration = duration ?? tracks.reduce((d, t) => Math.max(d, t.duration), 0);
  }
}
