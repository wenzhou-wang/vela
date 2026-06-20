import { ShaderPass, type ShaderPassInput, type ShaderPassOptions } from '../renderer/ShaderPass';
import type { WebGPURenderer } from '../renderer/WebGPURenderer';
import { AnimationStateMachine, type Condition, type TransitionDef } from '../animation/AnimationStateMachine';
import type { AnimationClip } from '../animation/AnimationClip';
import type { UniformValue } from '../materials/ShaderMaterial';

export interface AnimationGraphData {
  states: Array<{ name: string; clip?: string; blend?: { parameter: string; samples: Array<{ clip: string; position: number }> } }>;
  transitions: TransitionDef[];
  initial?: string;
  parameters?: Record<string, number>;
}

export interface AgentGraphSnapshot {
  post: Array<{ id: string; name: string; enabled: boolean; inputs: ShaderPassInput[]; effect: string; uniforms: Record<string, unknown> }>;
  animations: Record<string, { graph: AnimationGraphData; runtime: ReturnType<AnimationStateMachine['describe']> }>;
}

export interface AgentGraphChange { path: string; before?: unknown; after?: unknown }

export type AgentGraphEdit =
  | { type: 'post.add'; pass: ShaderPassOptions; index?: number }
  | { type: 'post.remove'; id: string }
  | { type: 'post.move'; id: string; index: number }
  | { type: 'post.update'; id: string; patch: { name?: string; enabled?: boolean; inputs?: ShaderPassInput[]; effect?: string; uniforms?: Record<string, UniformValue> } }
  | { type: 'animation.parameter'; machine: string; name: string; value: number }
  | { type: 'animation.play'; machine: string; state: string; duration?: number }
  | { type: 'animation.replace'; machine: string; graph: AnimationGraphData };

/** Structured read/write/diff surface over post and animation graphs for agents. */
export class AgentGraphEditor {
  private clips = new Map<string, AnimationClip>();

  constructor(
    private renderer: WebGPURenderer,
    private machines: Record<string, AnimationStateMachine> = {},
    clips: AnimationClip[] = [],
  ) {
    for (const clip of clips) this.registerClip(clip);
    for (const machine of Object.values(machines)) {
      for (const state of machine.getDefinition().states) {
        if (state.clip) this.registerClip(state.clip);
        for (const sample of state.blend?.samples ?? []) this.registerClip(sample.clip);
      }
    }
  }

  registerClip(clip: AnimationClip): this {
    if (!clip.name) throw new Error('AgentGraphEditor: clips need non-empty names.');
    const existing = this.clips.get(clip.name);
    if (existing && existing !== clip) throw new Error(`AgentGraphEditor: duplicate clip name "${clip.name}".`);
    this.clips.set(clip.name, clip);
    return this;
  }

  describe(): AgentGraphSnapshot {
    const animations: AgentGraphSnapshot['animations'] = {};
    for (const [id, machine] of Object.entries(this.machines)) {
      animations[id] = { graph: this.graphData(machine), runtime: machine.describe() };
    }
    return {
      post: this.renderer.passes.map((pass) => ({
        id: pass.id, name: pass.name, enabled: pass.enabled, inputs: [...pass.inputs],
        effect: pass.effectCode, uniforms: mapUniforms(pass.uniforms),
      })),
      animations,
    };
  }

  apply(edits: AgentGraphEdit | AgentGraphEdit[]): AgentGraphChange[] {
    const before = this.describe();
    for (const edit of Array.isArray(edits) ? edits : [edits]) this.applyOne(edit);
    return AgentGraphEditor.diff(before, this.describe());
  }

  static diff(before: AgentGraphSnapshot, after: AgentGraphSnapshot): AgentGraphChange[] {
    const changes: AgentGraphChange[] = [];
    diffValue(before, after, '', changes);
    return changes;
  }

  private applyOne(edit: AgentGraphEdit): void {
    if (edit.type === 'post.add') {
      const pass = new ShaderPass(edit.pass);
      const index = Math.max(0, Math.min(edit.index ?? this.renderer.passes.length, this.renderer.passes.length));
      this.renderer.passes.splice(index, 0, pass);
      return;
    }
    if (edit.type === 'post.remove' || edit.type === 'post.move' || edit.type === 'post.update') {
      const id = edit.id;
      const index = this.renderer.passes.findIndex((pass) => pass.id === id);
      if (index < 0) throw new Error(`AgentGraphEditor: unknown post pass "${id}".`);
      if (edit.type === 'post.remove') this.renderer.passes.splice(index, 1);
      else if (edit.type === 'post.move') {
        const [pass] = this.renderer.passes.splice(index, 1);
        this.renderer.passes.splice(Math.max(0, Math.min(edit.index, this.renderer.passes.length)), 0, pass);
      } else if (edit.type === 'post.update') {
        const pass = this.renderer.passes[index], patch = edit.patch;
        if (patch.name !== undefined) pass.name = patch.name;
        if (patch.enabled !== undefined) pass.enabled = patch.enabled;
        if (patch.inputs !== undefined) pass.inputs = [...patch.inputs];
        if (patch.uniforms !== undefined) pass.uniforms = patch.uniforms;
        if (patch.effect !== undefined) pass.setEffect(patch.effect);
      }
      return;
    }
    const machine = this.machines[edit.machine];
    if (!machine) throw new Error(`AgentGraphEditor: unknown animation machine "${edit.machine}".`);
    if (edit.type === 'animation.parameter') machine.setParameter(edit.name, edit.value);
    else if (edit.type === 'animation.play') machine.play(edit.state, edit.duration ?? 0);
    else if (edit.type === 'animation.replace') machine.setDefinition(this.resolveGraph(edit.graph));
  }

  private graphData(machine: AnimationStateMachine): AnimationGraphData {
    const def = machine.getDefinition();
    return {
      states: def.states.map((state) => ({
        name: state.name, clip: state.clip?.name,
        blend: state.blend ? { parameter: state.blend.parameter, samples: state.blend.samples.map((s) => ({ clip: s.clip.name, position: s.position })) } : undefined,
      })),
      transitions: def.transitions.map((t) => ({ ...t, when: t.when.map((c) => ({ ...c })) })),
      initial: def.initial,
      parameters: { ...def.parameters },
    };
  }

  private resolveGraph(graph: AnimationGraphData) {
    const clip = (name: string): AnimationClip => {
      const value = this.clips.get(name);
      if (!value) throw new Error(`AgentGraphEditor: unknown clip "${name}".`);
      return value;
    };
    return {
      states: graph.states.map((state) => ({
        name: state.name,
        clip: state.clip ? clip(state.clip) : undefined,
        blend: state.blend ? { parameter: state.blend.parameter, samples: state.blend.samples.map((s) => ({ clip: clip(s.clip), position: s.position })) } : undefined,
      })),
      transitions: graph.transitions.map((t) => ({ ...t, when: t.when.map((c: Condition) => ({ ...c })) })),
      initial: graph.initial,
      parameters: graph.parameters,
    };
  }
}

function mapUniforms(uniforms: Record<string, UniformValue>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(uniforms)) {
    if (typeof value === 'number') out[key] = value;
    else if ('isTexture' in value) out[key] = { texture: value.id, name: value.name };
    else if ('r' in value) out[key] = [value.r, value.g, value.b];
    else if ('w' in value) out[key] = [value.x, value.y, value.z, value.w];
    else if ('z' in value) out[key] = [value.x, value.y, value.z];
    else out[key] = [value.x, value.y];
  }
  return out;
}

function diffValue(before: unknown, after: unknown, path: string, out: AgentGraphChange[]): void {
  if (Object.is(before, after)) return;
  if (typeof before !== 'object' || before === null || typeof after !== 'object' || after === null) {
    out.push({ path: path || '/', before, after }); return;
  }
  const a = before as Record<string, unknown>, b = after as Record<string, unknown>;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) diffValue(a[key], b[key], `${path}/${key}`, out);
}
