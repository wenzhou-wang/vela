/**
 * Generates llms.txt — a single context-window-friendly page of vela's public
 * API — from the TypeScript surface of src/index.ts (plus src/test.ts).
 *
 *   node scripts/generate-llms.mjs
 *
 * Output: ./llms.txt and ./docs/llms.txt (same content).
 */
import ts from 'typescript';
import { writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const entry = join(root, 'src/index.ts');

const program = ts.createProgram([entry], {
  target: ts.ScriptTarget.ESNext,
  module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
  strict: true,
  types: ['@webgpu/types'],
});
const checker = program.getTypeChecker();
const source = program.getSourceFile(entry);
const moduleSymbol = checker.getSymbolAtLocation(source);
const exports = checker.getExportsOfModule(moduleSymbol);

const docOf = (symbol) => {
  const parts = symbol.getDocumentationComment(checker);
  const text = ts.displayPartsToString(parts).trim();
  if (!text) return '';
  // First paragraph, collapsed to one line.
  return text.split('\n\n')[0].replace(/\s+/g, ' ').trim();
};

const firstSentence = (text) => {
  if (!text) return '';
  const m = text.match(/^.*?[.!?](\s|$)/);
  return (m ? m[0] : text).trim();
};

const isPrivateName = (name) => name.startsWith('_') || name.startsWith('#');

function describeClass(symbol, decl) {
  const lines = [];
  const instanceType = checker.getDeclaredTypeOfSymbol(symbol);
  const heritage = decl.heritageClauses?.find((h) => h.token === ts.SyntaxKind.ExtendsKeyword);
  const ext = heritage ? ` extends ${heritage.types[0].getText()}` : '';
  lines.push(`### class ${symbol.name}${ext}`);
  const doc = docOf(symbol);
  if (doc) lines.push(doc);

  // Constructor
  const ctors = instanceType.getConstructSignatures?.() ?? [];
  const ctorSig = checker.getTypeOfSymbolAtLocation(symbol, decl).getConstructSignatures()[0];
  if (ctorSig && ctorSig.parameters.length > 0) {
    const params = ctorSig.parameters
      .map((p) => `${p.name}: ${checker.typeToString(checker.getTypeOfSymbolAtLocation(p, decl))}`)
      .join(', ');
    lines.push(`- new ${symbol.name}(${params})`);
  }

  for (const prop of instanceType.getProperties()) {
    if (isPrivateName(prop.name)) continue;
    const d = prop.declarations?.[0];
    if (!d) continue;
    const mods = ts.getCombinedModifierFlags(d);
    if (mods & (ts.ModifierFlags.Private | ts.ModifierFlags.Protected)) continue;
    // Only list members declared in this library (skip Object3D inherited spam
    // for subclasses — but keep them for Object3D itself).
    const declFile = d.getSourceFile().fileName;
    if (!declFile.includes('/src/')) continue;
    const ownerName = d.parent?.name?.getText?.();
    if (ownerName && ownerName !== symbol.name) continue; // inherited — listed on the base
    const type = checker.getTypeOfSymbolAtLocation(prop, decl);
    const sigs = type.getCallSignatures();
    const doc = firstSentence(docOf(prop));
    if (sigs.length > 0 && (ts.isMethodDeclaration(d) || ts.isMethodSignature(d))) {
      const s = sigs[0];
      const params = s.parameters
        .map((p) => `${p.name}${isOptional(p, d) ? '?' : ''}: ${shortType(checker.getTypeOfSymbolAtLocation(p, d))}`)
        .join(', ');
      lines.push(`- ${prop.name}(${params}): ${shortType(s.getReturnType())}${doc ? ` — ${doc}` : ''}`);
    } else {
      lines.push(`- ${prop.name}: ${shortType(type)}${doc ? ` — ${doc}` : ''}`);
    }
  }
  return lines.join('\n');
}

function isOptional(paramSymbol, ctx) {
  const d = paramSymbol.declarations?.[0];
  return d ? !!(d.questionToken || d.initializer) : false;
}

function shortType(type) {
  let s = checker.typeToString(type, undefined, ts.TypeFormatFlags.NoTruncation);
  if (s.length > 90) s = checker.typeToString(type); // fall back to truncated
  return s;
}

function describeFunction(symbol, decl) {
  const type = checker.getTypeOfSymbolAtLocation(symbol, decl);
  const sig = type.getCallSignatures()[0];
  if (!sig) return '';
  const params = sig.parameters
    .map((p) => `${p.name}${isOptional(p, decl) ? '?' : ''}: ${shortType(checker.getTypeOfSymbolAtLocation(p, decl))}`)
    .join(', ');
  const doc = docOf(symbol);
  return `### function ${symbol.name}(${params}): ${shortType(sig.getReturnType())}\n${doc}`;
}

function describeInterface(symbol, decl) {
  // Interfaces/type aliases print verbatim (they're the contract), comments stripped.
  const text = decl.getText().replace(/\/\*\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '')
    .split('\n').map((l) => l.trimEnd()).filter((l) => l.trim()).join('\n');
  const doc = docOf(symbol);
  return `### ${symbol.name}\n${doc ? doc + '\n' : ''}\`\`\`ts\n${text}\n\`\`\``;
}

const classes = [];
const functions = [];
const interfaces = [];

for (const exp of exports) {
  const symbol = exp.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(exp) : exp;
  const decl = symbol.declarations?.[0];
  if (!decl) continue;
  if (symbol.flags & ts.SymbolFlags.Class) classes.push(describeClass(symbol, decl));
  else if (symbol.flags & ts.SymbolFlags.Function) functions.push(describeFunction(symbol, decl));
  else if (symbol.flags & (ts.SymbolFlags.Interface | ts.SymbolFlags.TypeAlias)) {
    interfaces.push(describeInterface(symbol, decl));
  }
}

const readme = readFileSync(join(root, 'README.md'), 'utf8');
// First full paragraph after the title, unwrapped and de-markdowned.
const lines = readme.split('\n');
let start = lines.findIndex((l) => l.trim() && !l.startsWith('#'));
let para = [];
for (let i = start; i >= 0 && i < lines.length && lines[i].trim(); i++) para.push(lines[i].trim());
const tagline = para.join(' ').replace(/\*\*/g, '').replace(/\s+/g, ' ');

const out = `# vela

> ${tagline.trim() || 'An AI-first WebGPU rendering engine for the modern web.'}

vela is a WebGPU-only 3D rendering engine designed for AI agents building games:
declarative APIs, no hidden update flags, actionable error messages, deterministic
rendering, pixel readback, and built-in scene diagnostics. TypeScript, zero runtime
dependencies.

Quick start:

\`\`\`ts
import { WebGPURenderer, Scene, PerspectiveCamera, Mesh, BoxGeometry,
         StandardMaterial, DirectionalLight, Vector3 } from 'vela';

const renderer = new WebGPURenderer({ canvas });
await renderer.init();
const scene = new Scene();
scene.sky = { sunDirection: new Vector3(0.4, 0.7, 0.2) };  // procedural daylight + IBL
scene.skybox = true;
scene.add(new Mesh(new BoxGeometry(), new StandardMaterial({ color: 0x6699ff })));
const light = new DirectionalLight(0xffffff, 3);
scene.add(light, light.target);
const camera = new PerspectiveCamera(45, canvas.width / canvas.height, 0.1, 100);
camera.position.set(3, 2, 5); camera.lookAt(new Vector3(0, 0, 0));
function frame() { requestAnimationFrame(frame); renderer.render(scene, camera); }
frame();
\`\`\`

Stuck? \`renderer.diagnose(scene, camera)\` explains black screens with fixes;
\`describeScene(scene, camera)\` and \`renderer.report()\` return JSON state;
\`await renderer.screenshot()\` shows you the frame; \`expectFrame(...)\` turns
visuals into assertions. Docs: docs/GETTING_STARTED.md, docs/ARCHITECTURE.md,
docs/SHADER_MATERIAL.md, ROADMAP.md.

## Classes

${classes.join('\n\n')}

## Functions

${functions.join('\n\n')}

## Interfaces & types

${interfaces.join('\n\n')}
`;

writeFileSync(join(root, 'llms.txt'), out);
writeFileSync(join(root, 'docs/llms.txt'), out);
console.log(`llms.txt generated: ${out.length.toLocaleString()} chars, ${classes.length} classes, ${functions.length} functions, ${interfaces.length} interfaces`);
