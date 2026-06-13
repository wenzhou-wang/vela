import { FRAME_DEFS } from './pbr.wgsl';

/**
 * GPU particles.
 *
 * Simulation: one compute invocation per pool slot. Emission uses a ring
 * cursor — the CPU advances `cursor` by `count` each frame and slots inside
 * the wrapped window [cursor, cursor+count) respawn at the emitter; everything
 * else integrates (age += dt, velocity += gravity·dt, position += velocity·dt).
 * No atomics, no per-particle CPU work; a zeroed buffer means "all dead"
 * (lifetime 0).
 *
 * Rendering: one instanced draw of 6 vertices per pool slot. The vertex stage
 * builds a camera-facing quad from the view matrix basis, sizes/tints it by
 * normalized age, and emits a degenerate position for dead slots. The fragment
 * is a soft disc, tonemapped exactly like the material path.
 */

export const PARTICLE_SIM_SHADER = /* wgsl */ `
struct Particle {
  posAge  : vec4<f32>,  // xyz = position, w = age (s)
  velLife : vec4<f32>,  // xyz = velocity, w = lifetime (s; 0 = dead)
  seed    : vec4<f32>,  // x = per-particle random 0..1
};

struct SimParams {
  emitterPos : vec4<f32>,  // xyz = world emit position, w = dt
  velocity   : vec4<f32>,  // xyz = base velocity, w = spread
  gravity    : vec4<f32>,  // xyz = acceleration, w = time (random seed)
  life       : vec4<f32>,  // x = min lifetime, y = max lifetime
  emit       : vec4<u32>,  // x = ring cursor, y = emit count, z = capacity
};

@group(0) @binding(0) var<uniform> p : SimParams;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;

fn pcg(n : u32) -> u32 {
  var h = n * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

fn rand(n : u32) -> f32 {
  return f32(pcg(n)) / 4294967295.0;
}

@compute @workgroup_size(64)
fn cs_sim(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let cap = p.emit.z;
  if (i >= cap) { return; }

  // Ring-buffer emission window (wraps at capacity).
  let rel = (i + cap - (p.emit.x % cap)) % cap;
  if (rel < p.emit.y) {
    let base = i * 4u + u32(p.gravity.w * 1024.0) * 2654435761u;
    let r1 = rand(base);
    let r2 = rand(base + 1u);
    let r3 = rand(base + 2u);
    let r4 = rand(base + 3u);
    var out : Particle;
    out.posAge = vec4<f32>(p.emitterPos.xyz, 0.0);
    out.velLife = vec4<f32>(
      p.velocity.xyz + (vec3<f32>(r1, r2, r3) * 2.0 - 1.0) * p.velocity.w,
      mix(p.life.x, p.life.y, r4),
    );
    out.seed = vec4<f32>(r1, 0.0, 0.0, 0.0);
    particles[i] = out;
    return;
  }

  var part = particles[i];
  if (part.velLife.w <= 0.0 || part.posAge.w >= part.velLife.w) { return; }
  let dt = p.emitterPos.w;
  part.posAge.w = part.posAge.w + dt;
  let vel = part.velLife.xyz + p.gravity.xyz * dt;
  part.velLife = vec4<f32>(vel, part.velLife.w);
  part.posAge = vec4<f32>(part.posAge.xyz + vel * dt, part.posAge.w);
  particles[i] = part;
}
`;

export const PARTICLE_DRAW_SHADER = FRAME_DEFS + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame : Frame;

struct Particle {
  posAge  : vec4<f32>,
  velLife : vec4<f32>,
  seed    : vec4<f32>,
};

struct DrawParams {
  sizeRange : vec4<f32>,  // x = start size, y = end size (world units)
  color0    : vec4<f32>,  // rgb + opacity at birth
  color1    : vec4<f32>,  // rgb + opacity at death
};

@group(1) @binding(0) var<storage, read> particles : array<Particle>;
@group(1) @binding(1) var<uniform> dp : DrawParams;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var out : VSOut;
  let part = particles[ii];
  let life = part.velLife.w;
  if (life <= 0.0 || part.posAge.w >= life) {
    out.clip = vec4<f32>(0.0, 0.0, -2.0, 1.0); // clipped away
    return out;
  }
  let t = clamp(part.posAge.w / life, 0.0, 1.0);
  // Quad corners (two CCW triangles); var so the runtime index is valid WGSL.
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  let c = corners[vi];
  let halfSize = mix(dp.sizeRange.x, dp.sizeRange.y, t) * 0.5;
  // Camera basis from the view matrix rows (world-space right/up).
  let right = vec3<f32>(frame.view[0].x, frame.view[1].x, frame.view[2].x);
  let up    = vec3<f32>(frame.view[0].y, frame.view[1].y, frame.view[2].y);
  let world = part.posAge.xyz + (right * c.x + up * c.y) * halfSize;
  out.clip = frame.proj * frame.view * vec4<f32>(world, 1.0);
  out.uv = c;
  out.color = mix(dp.color0, dp.color1, t);
  return out;
}

fn pAcesFilmic(x : vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn pLinearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  return mix(c * 12.92, 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055, cutoff);
}

// Soft disc; premultiplied output so one pipeline blend state per mode works:
// additive (one, one) and alpha (one, one-minus-src-alpha) both expect
// alpha-weighted color.
@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  let alpha = in.color.a * smoothstep(1.0, 0.55, d);
  if (alpha <= 0.001) { discard; }
  var rgb = in.color.rgb * frame.ambient.w; // exposure
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    rgb = pAcesFilmic(rgb);
    rgb = pLinearToSRGB(rgb);
  }
  return vec4<f32>(rgb * alpha, alpha);
}
`;
