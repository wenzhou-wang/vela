/**
 * Procedural sky generation: evaluates the Preetham analytic daylight model
 * ("A Practical Analytic Model for Daylight", Preetham et al. 1999) into an
 * equirectangular rgba16float texture. The texture then feeds the same paths
 * a loaded HDR environment would: the IBL prefilter and the skybox pass.
 *
 * Direction mapping matches dirToEquirectUv in pbr.wgsl:
 *   u = atan2(z, x) / 2pi + 0.5,  v = acos(y) / pi.
 */
export const SKYGEN_SHADER = /* wgsl */ `
const PI = 3.141592653589793;

struct SkyGenParams {
  sunDir : vec4<f32>,  // xyz = normalized direction toward the sun, w = turbidity
};

@group(0) @binding(0) var<uniform> p : SkyGenParams;
@group(0) @binding(1) var outTex : texture_storage_2d<rgba16float, write>;

// Perez sky luminance distribution.
fn perez(cosTheta : f32, gamma : f32, cosGamma : f32, c : array<f32, 5>) -> f32 {
  return (1.0 + c[0] * exp(c[1] / max(cosTheta, 0.01))) *
         (1.0 + c[2] * exp(c[3] * gamma) + c[4] * cosGamma * cosGamma);
}

// CIE xyY -> linear sRGB.
fn xyYToRGB(x : f32, y : f32, Y : f32) -> vec3<f32> {
  let X = x / max(y, 1e-4) * Y;
  let Z = (1.0 - x - y) / max(y, 1e-4) * Y;
  return vec3<f32>(
     3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
     0.0557 * X - 0.2040 * Y + 1.0570 * Z,
  );
}

@compute @workgroup_size(8, 8)
fn cs_sky(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(outTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
  let phi = (uv.x - 0.5) * 2.0 * PI;
  let theta = uv.y * PI;
  let dir = vec3<f32>(cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta));

  let sun = normalize(p.sunDir.xyz);
  let T = clamp(p.sunDir.w, 1.7, 10.0);
  let thetaS = acos(clamp(sun.y, -1.0, 1.0));

  // Perez coefficients (Y / x / y) as linear functions of turbidity.
  let cY = array<f32, 5>( 0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251,
                          0.1206 * T - 2.5771, -0.0670 * T + 0.3703);
  let cx = array<f32, 5>(-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125,
                         -0.0641 * T - 0.8989, -0.0033 * T + 0.0452);
  let cy = array<f32, 5>(-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102,
                         -0.0441 * T - 1.6537, -0.0109 * T + 0.0529);

  // Zenith values.
  let chi = (4.0 / 9.0 - T / 120.0) * (PI - 2.0 * thetaS);
  let Yz = max((4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192, 0.001); // kcd/m^2
  let t2 = T * T;
  let s = thetaS;
  let s2 = s * s;
  let s3 = s2 * s;
  let xz = t2 * (0.00166 * s3 - 0.00375 * s2 + 0.00209 * s) +
           T  * (-0.02903 * s3 + 0.06377 * s2 - 0.03202 * s + 0.00394) +
                ( 0.11693 * s3 - 0.21196 * s2 + 0.06052 * s + 0.25886);
  let yz = t2 * (0.00275 * s3 - 0.00610 * s2 + 0.00317 * s) +
           T  * (-0.04214 * s3 + 0.08970 * s2 - 0.04153 * s + 0.00516) +
                ( 0.15346 * s3 - 0.26756 * s2 + 0.06670 * s + 0.26688);

  // Evaluate the model just above the horizon for downward rays, then fade.
  let cosTheta = max(dir.y, 0.01);
  let cosGamma = clamp(dot(dir, sun), -1.0, 1.0);
  let gamma = acos(cosGamma);
  let cosGammaS = clamp(sun.y, -1.0, 1.0); // gamma at zenith = thetaS

  let Y = Yz * perez(cosTheta, gamma, cosGamma, cY) / perez(1.0, thetaS, cosGammaS, cY);
  let x = xz * perez(cosTheta, gamma, cosGamma, cx) / perez(1.0, thetaS, cosGammaS, cx);
  let y = yz * perez(cosTheta, gamma, cosGamma, cy) / perez(1.0, thetaS, cosGammaS, cy);

  // Normalize to a ~1.0 mid-day scale (Yz is in kcd/m^2; /8 lands typical
  // zeniths near 1-3 in linear HDR, comfortable for the ACES tonemap).
  var rgb = max(xyYToRGB(x, y, Y / 8.0), vec3<f32>(0.0));

  // Sun disc (~0.27 deg radius) with a soft limb, only above the horizon.
  if (sun.y > 0.0) {
    let sunDisc = smoothstep(0.999956, 0.999989, cosGamma); // cos(0.54deg)..cos(0.27deg)
    rgb = rgb + vec3<f32>(50.0) * sunDisc * smoothstep(0.0, 0.05, dir.y);
  }

  // Below the horizon: fade to a dim ground bounce tint.
  if (dir.y < 0.0) {
    let fade = smoothstep(-0.1, 0.0, dir.y);
    let ground = vec3<f32>(0.31, 0.27, 0.24) * (Yz / 8.0) * 0.12;
    rgb = mix(ground, rgb, fade);
  }

  textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}
`;
