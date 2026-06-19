import type { AnimationClip } from './AnimationClip';
import type { AnimationAction } from './AnimationAction';
import { AnimationMixer } from './AnimationMixer';

/** A comparison on a named parameter; a transition fires when all of its conditions hold. */
export interface Condition {
  parameter: string;
  op: '<' | '<=' | '>' | '>=' | '==' | '!=';
  value: number;
}

/** A 1-D blend-space sample: a clip placed at `position` along the blend parameter. */
export interface BlendSample {
  clip: AnimationClip;
  position: number;
}

/**
 * A graph node: either a single `clip`, or a 1-D `blend` space that mixes its
 * samples by the value of `parameter` (nearest two, linearly weighted).
 */
export interface StateDef {
  name: string;
  clip?: AnimationClip;
  blend?: { parameter: string; samples: BlendSample[] };
}

/** `from → to` when every condition in `when` holds; `from: '*'` matches any state. */
export interface TransitionDef {
  from: string;
  to: string;
  when: Condition[];
  /** Crossfade seconds (0 = instant). */
  duration?: number;
}

export interface StateMachineDef {
  states: StateDef[];
  transitions: TransitionDef[];
  /** Starting state (defaults to the first). */
  initial?: string;
  /** Initial parameter values. */
  parameters?: Record<string, number>;
}

interface ResolvedState {
  def: StateDef;
  samples: BlendSample[]; // sorted by position (empty for a single-clip state)
}

/**
 * A declarative animation state machine over an `AnimationMixer`. States are
 * single clips or 1-D blend spaces; transitions fire on parameter conditions
 * and crossfade. Each frame it sets the mixer's per-clip action weights, so all
 * the blending/cross-fade machinery is reused. Introspect with `describe()`.
 *
 * Data-first by design: states, transitions, and parameters are plain values an
 * agent can build, mutate (`setParameter`), and read back without a UI.
 */
export class AnimationStateMachine {
  readonly mixer = new AnimationMixer();
  readonly parameters: Record<string, number>;
  current: string;

  private states = new Map<string, ResolvedState>();
  private transitions: TransitionDef[];
  private actions = new Map<AnimationClip, AnimationAction>();
  // Active crossfade: blending `from` → `current` over `dur` seconds.
  private from: string | null = null;
  private dur = 0;
  private elapsed = 0;

  constructor(def: StateMachineDef) {
    for (const s of def.states) {
      const samples = s.blend ? [...s.blend.samples].sort((a, b) => a.position - b.position) : [];
      this.states.set(s.name, { def: s, samples });
    }
    this.transitions = def.transitions;
    this.parameters = { ...(def.parameters ?? {}) };
    this.current = def.initial ?? def.states[0].name;
  }

  setParameter(name: string, value: number): this {
    this.parameters[name] = value;
    return this;
  }

  /** Immediately switch to `name` with an optional crossfade (no condition check). */
  play(name: string, duration = 0): this {
    this.beginTransition(name, duration);
    return this;
  }

  /** Evaluate transitions, accumulate state weights into the mixer, and advance it. */
  update(dt: number): void {
    if (this.from === null) {
      for (const tr of this.transitions) {
        if (tr.from !== this.current && tr.from !== '*') continue;
        if (tr.to === this.current) continue;
        if (this.conditionsMet(tr.when)) { this.beginTransition(tr.to, tr.duration ?? 0); break; }
      }
    }

    if (this.from !== null) {
      this.elapsed += dt;
      if (this.elapsed >= this.dur) { this.from = null; this.elapsed = 0; }
    }

    // Reset, then accumulate each contributing state's weights (ratios only —
    // the mixer normalizes, so a crossfade is just (1-s) old + s new).
    for (const a of this.actions.values()) a.weight = 0;
    if (this.from !== null) {
      const s = this.dur > 0 ? this.elapsed / this.dur : 1;
      this.addStateWeights(this.from, 1 - s);
      this.addStateWeights(this.current, s);
    } else {
      this.addStateWeights(this.current, 1);
    }

    this.mixer.update(dt);
  }

  private beginTransition(to: string, duration: number): void {
    if (!this.states.has(to)) return;
    if (duration > 0 && to !== this.current) {
      this.from = this.current;
      this.dur = duration;
      this.elapsed = 0;
    } else {
      this.from = null;
    }
    this.current = to;
  }

  private conditionsMet(when: Condition[]): boolean {
    for (const c of when) {
      const v = this.parameters[c.parameter] ?? 0;
      let ok: boolean;
      switch (c.op) {
        case '<': ok = v < c.value; break;
        case '<=': ok = v <= c.value; break;
        case '>': ok = v > c.value; break;
        case '>=': ok = v >= c.value; break;
        case '==': ok = v === c.value; break;
        case '!=': ok = v !== c.value; break;
      }
      if (!ok) return false;
    }
    return true;
  }

  /** Add a state's clip weights (scaled by `scale`) into the managed actions. */
  private addStateWeights(name: string, scale: number): void {
    const st = this.states.get(name);
    if (!st || scale <= 0) return;
    if (st.def.clip) {
      this.action(st.def.clip).weight += scale;
      return;
    }
    const samples = st.samples;
    if (samples.length === 0) return;
    const p = this.parameters[st.def.blend!.parameter] ?? 0;
    if (p <= samples[0].position) {
      this.action(samples[0].clip).weight += scale;
    } else if (p >= samples[samples.length - 1].position) {
      this.action(samples[samples.length - 1].clip).weight += scale;
    } else {
      let i = 0;
      while (i < samples.length - 1 && samples[i + 1].position <= p) i++;
      const a = (p - samples[i].position) / (samples[i + 1].position - samples[i].position);
      this.action(samples[i].clip).weight += scale * (1 - a);
      this.action(samples[i + 1].clip).weight += scale * a;
    }
  }

  /** Persistent (one per clip) action on the mixer, created on first use. */
  private action(clip: AnimationClip): AnimationAction {
    let a = this.actions.get(clip);
    if (!a) {
      a = this.mixer.add(clip, 0);
      this.actions.set(clip, a);
    }
    return a;
  }

  /** JSON-friendly snapshot: current state, parameters, transition, active clips. */
  describe(): {
    current: string;
    parameters: Record<string, number>;
    transition: { from: string; to: string; progress: number } | null;
    active: { clip: string; weight: number }[];
  } {
    let total = 0;
    for (const a of this.actions.values()) total += a.weight > 0 ? a.weight : 0;
    const active: { clip: string; weight: number }[] = [];
    for (const a of this.actions.values()) {
      if (a.weight > 0) active.push({ clip: a.clip.name, weight: total > 0 ? a.weight / total : 0 });
    }
    return {
      current: this.current,
      parameters: { ...this.parameters },
      transition: this.from !== null
        ? { from: this.from, to: this.current, progress: this.dur > 0 ? this.elapsed / this.dur : 1 }
        : null,
      active,
    };
  }
}
