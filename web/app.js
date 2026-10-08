import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const COL = { slide: 0x0d131c, neuropil: new THREE.Color('#c2408f'), spike: new THREE.Color('#63e68c'),
  sense: new THREE.Color('#8fb8e8'), act: new THREE.Color('#f2a93b') };

const state = { replay: null, idx: 0, playing: false, speed: 1, simT: 0, follow: true, last: performance.now() };

// ------------------------------------------------------------------ arena view
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(COL.slide);
scene.fog = new THREE.Fog(COL.slide, 22, 60);
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2; // MuJoCo is z-up; three.js is y-up
scene.add(world);
const camera = new THREE.PerspectiveCamera(48, 1, 0.02, 200);
camera.position.set(-2, 2, 3);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a2f38, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-6, 14, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, near: 1, far: 50 });
scene.add(sun, sun.target);

const arenaGroup = new THREE.Group();
world.add(arenaGroup);

function checker(c1, c2, n = 8, size = 256) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const g = cv.getContext('2d');
  const s = size / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    g.fillStyle = (i + j) % 2 ? c1 : c2;
    g.fillRect(i * s, j * s, s, s);
  }
  const t = new THREE.CanvasTexture(cv);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const std = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0.05, ...extra });

const PAD = { sugar: 0xffcc26, bitter: 0x40bf4d, mixed: 0xf2731a, plain: 0x9a9a9a };
let loomMeshes = [];
let puffMesh = null;
let trail = null;

function buildArena(r) {
  arenaGroup.clear();
  loomMeshes = [];
  const a = r.arena;
  const [hx, hy] = a.size, [cx, cy] = a.center;
  const ftex = checker('#4a4f57', '#5c626b', 2, 128);
  ftex.repeat.set(hx * 2, hy * 2);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(2 * hx, 2 * hy), std(0xffffff, { map: ftex, roughness: 0.95 }));
  floor.position.set(cx, cy, 0);
  floor.receiveShadow = true;
  arenaGroup.add(floor);
  if (a.walls) {
    const h = a.wall_height, t = 0.05;
    const wt = checker('#e9e9e6', '#6b707a', 2, 128);
    const mk = (w, d, x, y, rx, ry) => {
      const tx = wt.clone(); tx.needsUpdate = true; tx.repeat.set(rx, ry);
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), std(0xffffff, { map: tx }));
      m.position.set(x, y, h / 2); m.receiveShadow = true; arenaGroup.add(m);
    };
    mk(2 * hx + 4 * t, 2 * t, cx, cy + hy + t, 2 * hx, h);
    mk(2 * hx + 4 * t, 2 * t, cx, cy - hy - t, 2 * hx, h);
    mk(2 * t, 2 * hy, cx + hx + t, cy, 2 * hy, h);
    mk(2 * t, 2 * hy, cx - hx - t, cy, 2 * hy, h);
  }
  const otex = checker('#c23a2e', '#f2ebe0', 4, 64);
  for (const o of a.objects) {
    const [x, y] = o.pos;
    if (o.kind === 'cylinder') {
      const [rad, hh] = o.size;
      const tx = otex.clone(); tx.needsUpdate = true; tx.repeat.set(3, 3);
      const m = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, 2 * hh, 32).rotateX(Math.PI / 2), std(0xffffff, { map: tx }));
      m.position.set(x, y, hh); m.castShadow = m.receiveShadow = true; arenaGroup.add(m);
    } else if (o.kind === 'box') {
      const [sx, sy, sz] = o.size;
      const m = new THREE.Mesh(new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz), std(0xc23a2e));
      m.position.set(x, y, sz); m.rotation.z = o.yaw || 0; m.castShadow = true; arenaGroup.add(m);
    } else if (o.kind === 'beacon') {
      const h = o.height ?? 0.35, rad = o.radius ?? 0.18;
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, h, 12).rotateX(Math.PI / 2), std(0x5a5a60));
      post.position.set(x, y, h / 2); arenaGroup.add(post);
      const glow = new THREE.Mesh(new THREE.SphereGeometry(rad, 32, 16), new THREE.MeshStandardMaterial({ color: 0xfff2a0, emissive: 0xfff0b0, emissiveIntensity: 1.4 }));
      glow.position.set(x, y, h + rad); arenaGroup.add(glow);
      const pl = new THREE.PointLight(0xffe6a0, 2.2, 4, 1.6); pl.position.copy(glow.position); arenaGroup.add(pl);
    } else if (o.kind === 'pad') {
      const [sx, sy] = o.size || [0.35, 0.35];
      const m = new THREE.Mesh(new THREE.BoxGeometry(2 * sx, 2 * sy, 0.004), std(PAD[o.taste] ?? 0x999999, { roughness: 0.5 }));
      m.position.set(x, y, 0.002); m.receiveShadow = true; arenaGroup.add(m);
    } else if (o.kind === 'looming') {
      const m = new THREE.Mesh(new THREE.SphereGeometry(o.radius ?? 0.25, 32, 16), std(0x050505, { roughness: 0.9 }));
      m.castShadow = true; m.position.set(50, 50, -5); arenaGroup.add(m); loomMeshes.push(m);
    } else if (o.kind === 'source') {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), std(0xf2d933, { emissive: 0x5a4a00 }));
      m.scale.set(0.1, 0.06, 0.08); m.position.set(x, y, 0.08); arenaGroup.add(m);
    } else if (o.kind === 'gate') {
      const w = o.width ?? 1.2, h = o.height ?? 1.6;
      const g = new THREE.Group(); g.position.set(x, y, 0); g.rotation.z = o.yaw || 0;
      const mat = std(0x26a6f2, { emissive: 0x0a2a40 });
      for (const s of [-1, 1]) {
        const p = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, h, 12).rotateX(Math.PI / 2), mat);
        p.position.set(0, s * w / 2, h / 2); g.add(p);
      }
      const top = new THREE.Mesh(new THREE.BoxGeometry(0.08, w, 0.08), mat); top.position.set(0, 0, h); g.add(top);
      arenaGroup.add(g);
    }
  }
  puffMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.035, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xf4e04d, transparent: true, opacity: 0.35, depthWrite: false }), 600);
  puffMesh.count = 0;
  arenaGroup.add(puffMesh);
  const tg = new THREE.BufferGeometry();
  tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(r.frames.length, 9000) * 3), 3));
  tg.setDrawRange(0, 0);
  trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0xf2a93b, transparent: true, opacity: 0.55 }));
  arenaGroup.add(trail);
  r.frames.forEach((f, i) => trailPoint(i, f));
}
function trailPoint(i, f) {
  const P = trail.geometry.attributes.position.array;
  if (3 * i + 2 >= P.length) return;
  P[3 * i] = f.pos[0]; P[3 * i + 1] = f.pos[1]; P[3 * i + 2] = Math.max(f.pos[2] - 0.07, 0.01);
  trail.geometry.attributes.position.needsUpdate = true;
}

// ------------------------------------------------------------------ robot model
const robot = { root: new THREE.Group(), wheels: {}, props: [], ready: false };
world.add(robot.root);

function geomMesh(p) {
  const [a, b, c] = p.size;
  let g;
  if (p.type === 'box') g = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
  else if (p.type === 'cylinder') g = new THREE.CylinderGeometry(a, a, 2 * b, 28).rotateX(Math.PI / 2);
  else if (p.type === 'sphere') g = new THREE.SphereGeometry(a, 20, 12);
  else if (p.type === 'capsule') g = new THREE.CapsuleGeometry(a, 2 * b, 6, 12).rotateX(Math.PI / 2);
  else if (p.type === 'ellipsoid') g = new THREE.SphereGeometry(1, 20, 12).scale(a, b, c);
  const [r, gr, bl, al] = p.rgba;
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, gr, bl), roughness: 0.55, metalness: 0.25,
    transparent: al < 1, opacity: al, emissive: new THREE.Color(r, gr, bl).multiplyScalar(p.emission || 0),
  });
  const m = new THREE.Mesh(g, mat);
  m.position.fromArray(p.pos);
  m.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
  m.castShadow = al >= 1;
  return m;
}

async function buildRobot() {
  const R = await (await fetch('assets/robot.json')).json();
  const groups = { robot: robot.root };
  for (const [name, b] of Object.entries(R.bodies)) {
    const g = new THREE.Group();
    g.position.fromArray(b.pos);
    g.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    g.userData.base = g.quaternion.clone();
    robot.root.add(g);
    groups[name] = g;
    if (name.startsWith('wheel_')) robot.wheels[name.slice(6)] = g;
  }
  for (const p of R.parts) {
    if (p.name.startsWith('prop_')) {
      const g = new THREE.Group();
      g.position.fromArray(p.pos);
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(p.size[0], p.size[0], 0.001, 48).rotateX(Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(...p.rgba.slice(0, 3)), transparent: true, opacity: 0.16, depthWrite: false }));
      g.add(disc);
      const blade = new THREE.Mesh(new THREE.BoxGeometry(2 * p.size[0] * 0.97, 0.016, 0.003),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(...p.rgba.slice(0, 3)), roughness: 0.4 }));
      const blades = new THREE.Group(); blades.add(blade); g.add(blades);
      robot.root.add(g);
      robot.props.push({ blades, disc, spin: p.name.endsWith('FL') || p.name.endsWith('RR') ? 1 : -1 });
      continue;
    }
    (groups[p.body] ?? robot.root).add(geomMesh(p));
  }
  robot.ready = true;
}

const _q = new THREE.Quaternion(), _axisY = new THREE.Vector3(0, 1, 0);
function poseRobot(f) {
  robot.root.position.fromArray(f.pos);
  robot.root.quaternion.set(f.quat[1], f.quat[2], f.quat[3], f.quat[0]);
  const names = ['FL', 'FR', 'RL', 'RR'];
  names.forEach((n, i) => {
    const w = robot.wheels[n];
    if (w) w.quaternion.copy(w.userData.base).multiply(_q.setFromAxisAngle(_axisY, f.wheels[i]));
  });
  robot.props.forEach((p, i) => {
    const thr = f.thrust[i] || 0;
    p.disc.material.opacity = thr > 0.05 ? 0.22 : 0.08;
    if (thr > 0.05 && !reduceMotion) p.blades.rotation.z += p.spin * 0.9;
  });
}

// ------------------------------------------------------------------ brain view
const bcanvas = $('brain');
const brenderer = new THREE.WebGLRenderer({ canvas: bcanvas, antialias: true });
brenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const bscene = new THREE.Scene();
bscene.background = new THREE.Color(COL.slide);
const bcam = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
bcam.position.set(0, 0, 4.2);
const brain = { pts: null, act: null, n: 0, pivot: new THREE.Group() };
bscene.add(brain.pivot);

async function buildBrain() {
  const [meta, xyzBuf, clsBuf] = await Promise.all([
    fetch('assets/brain.json').then((r) => r.json()),
    fetch('assets/brain.bin').then((r) => r.arrayBuffer()),
    fetch('assets/brain_class.bin').then((r) => r.arrayBuffer()),
  ]);
  const q = new Int16Array(xyzBuf);
  const n = meta.n;
  const pos = new Float32Array(n * 3);
  // fit the brain's width (robust 1st-99th percentile of x) to ~2.9 scene units
  const xs = Float32Array.from({ length: n }, (_, i) => q[3 * i]).sort();
  const s = 2.9 / Math.max(1, xs[Math.floor(n * 0.99)] - xs[Math.floor(n * 0.01)]);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = q[3 * i] * s;
    pos[3 * i + 1] = -q[3 * i + 1] * s;
    pos[3 * i + 2] = -q[3 * i + 2] * s;
  }
  const role = new Float32Array(n);
  for (const p of Object.values(meta.populations)) for (const i of p.idx) role[i] = p.role === 'input' ? 1 : 2;
  const act = new Float32Array(n);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('aRole', new THREE.BufferAttribute(role, 1));
  g.setAttribute('aAct', new THREE.BufferAttribute(act, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uPx: { value: brenderer.getPixelRatio() }, cAnat: { value: COL.neuropil }, cIn: { value: COL.sense }, cOut: { value: COL.act }, cSpike: { value: COL.spike } },
    vertexShader: `attribute float aRole; attribute float aAct; uniform float uPx; varying float vRole; varying float vAct;
      void main(){ vRole=aRole; vAct=aAct; vec4 mv=modelViewMatrix*vec4(position,1.0);
        gl_PointSize = uPx*(1.6 + aAct*5.0 + step(0.5,aRole)*1.4) * (3.6 / -mv.z); gl_Position=projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 cAnat; uniform vec3 cIn; uniform vec3 cOut; uniform vec3 cSpike; varying float vRole; varying float vAct;
      void main(){ vec2 d=gl_PointCoord-0.5; float r=dot(d,d); if(r>0.25) discard; float fall=1.0-r*4.0;
        vec3 base = vRole>1.5 ? cOut : (vRole>0.5 ? cIn : cAnat); float a = vRole>0.5 ? 0.6 : 0.16;
        vec3 col = mix(base, cSpike, clamp(vAct,0.0,1.0)); float alpha = max(a, vAct*0.95)*fall;
        gl_FragColor = vec4(col*alpha, alpha); }`,
  });
  brain.pts = new THREE.Points(g, mat);
  brain.pivot.add(brain.pts);
  brain.act = act;
  brain.n = n;
  brain.meta = meta;
  $('brainstats').textContent = `${n.toLocaleString()} neurons, ${meta.n_connections.toLocaleString()} connections`;
}

function b64ToU32(s) {
  const bin = atob(s);
  const b = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
  return new Uint32Array(b.buffer);
}

let lastBrainIdx = -1;
function brainFrame(i) {
  if (!brain.pts) return;
  const act = brain.act;
  if (i < lastBrainIdx || i - lastBrainIdx > 25) act.fill(0);
  const frames = state.replay.frames;
  for (let k = Math.max(lastBrainIdx + 1, i - 25); k <= i; k++) {
    for (let j = 0; j < brain.n; j++) act[j] *= 0.55;
    const b = frames[k] && frames[k].brain;
    if (b && b.act) { for (const v of b64ToU32(b.act)) act[v] = 1.0; }
  }
  lastBrainIdx = i;
  brain.pts.geometry.attributes.aAct.needsUpdate = true;
}

// ------------------------------------------------------------------ pathway strip
const PATHWAYS = [
  { name: 'Looming', sub: 'dark object expanding', feat: ['loom_L', 'loom_R'], fu: '1/s', inn: ['looming_L', 'looming_R'], whoIn: 'LPLC2 + LC4',
    out: ['GF_L', 'GF_R'], whoOut: 'Giant Fiber (DNp01)', act: (f) => (f.cmd.escape ? '<b>Escape: take off</b>' : 'no escape') },
  { name: 'Obstacle ahead', sub: 'approach rate (ToF ranging)', feat: ['approach_L', 'approach_R'], fu: '1/s', inn: ['obstacle_L', 'obstacle_R'], whoIn: 'LC16',
    out: ['DNa02_L', 'DNa02_R'], whoOut: 'DNa02 steering', act: turnText },
  { name: 'Beacon', sub: 'bright target in view', feat: ['target_drive_L', 'target_drive_R'], fu: '', inn: ['target_L', 'target_R'], whoIn: 'LC10a',
    out: ['DNa01_L', 'DNa01_R'], whoOut: 'DNa01 steering', act: turnText },
  { name: 'Rotation', sub: 'horizontal optic flow', feat: ['hs_L', 'hs_R'], fu: '', inn: ['yawflow_HS_L', 'yawflow_HS_R'], whoIn: 'HS cells (+ H2)',
    out: ['DNp15_L', 'DNp15_R'], whoOut: 'DNp15 / DNHS1', act: turnText },
  { name: 'Taste pad', sub: 'sugar | bitter contact', feat: ['taste_sugar', 'taste_bitter'], fu: '', inn: ['taste_sugar', 'taste_bitter'], whoIn: 'sugar | bitter GRNs',
    out: ['MN9_L', 'MN9_R'], whoOut: 'MN9 proboscis motor neuron', act: (f) => (f.cmd.halt ? '<b>Stop and dock ("feed")</b>' : 'keep moving') },
  { name: 'Vibration', sub: 'sound / substrate', feat: ['vibration_L', 'vibration_R'], fu: '', inn: ['vibration_L', 'vibration_R'], whoIn: 'Johnston\'s organ A/B',
    out: ['GF_L', 'GF_R'], whoOut: 'Giant Fiber (DNp01)', act: (f) => (f.cmd.escape ? '<b>Escape: take off</b>' : 'no escape') },
];
function turnText(f) {
  const y = f.cmd.yaw;
  if (Math.abs(y) < 0.05) return `straight, ${f.cmd.v.toFixed(2)} m/s`;
  return `<b>turn ${y > 0 ? 'left' : 'right'}</b> ${Math.abs(y).toFixed(2)} rad/s`;
}

function buildPathways() {
  const host = $('pathways');
  host.innerHTML = '';
  for (const p of PATHWAYS) {
    const row = document.createElement('div');
    row.className = 'pw quiet';
    row.innerHTML = `<div class="sense"><b>${p.name}</b><small>${p.sub}</small></div>
      <div class="neurons in"><span class="who">${p.whoIn}</span><div class="bars"><div class="bar l"><i></i><span></span></div><div class="bar r"><i></i><span></span></div></div></div>
      <div class="neurons out"><span class="who">${p.whoOut}</span><div class="bars"><div class="bar l"><i></i><span></span></div><div class="bar r"><i></i><span></span></div></div></div>
      <div class="act"></div>`;
    p.el = row;
    p.bars = row.querySelectorAll('.bar');
    p.actEl = row.querySelector('.act');
    host.appendChild(row);
  }
}

function setBar(bar, hz, max) {
  bar.querySelector('i').style.width = `${Math.min(100, (100 * hz) / max)}%`;
  bar.querySelector('span').textContent = hz >= 0.5 ? Math.round(hz) : '';
}

function updatePathways(f) {
  const b = f.brain;
  for (const p of PATHWAYS) {
    let hot = false;
    if (b) {
      const i0 = b.in[p.inn[0]] ?? 0, i1 = b.in[p.inn[1]] ?? 0;
      const o0 = b.dn[p.out[0]] ?? 0, o1 = b.dn[p.out[1]] ?? 0;
      setBar(p.bars[0], i0, 150); setBar(p.bars[1], i1, 150);
      setBar(p.bars[2], o0, 120); setBar(p.bars[3], o1, 120);
      hot = i0 + i1 > 1;
    } else {
      p.bars.forEach((x) => setBar(x, 0, 1));
      const v0 = f.feat[p.feat[0]] ?? 0, v1 = f.feat[p.feat[1]] ?? 0;
      hot = Math.abs(v0) + Math.abs(v1) > 1e-3;
    }
    p.el.classList.toggle('quiet', !hot);
    p.el.classList.toggle('hot', hot);
    p.actEl.innerHTML = hot ? p.act(f) : '';
  }
}

// ------------------------------------------------------------------ playback
let fpvIdx = -1;
function show(i) {
  const r = state.replay;
  if (!r) return;
  i = Math.max(0, Math.min(r.frames.length - 1, i));
  state.idx = i;
  const f = r.frames[i];
  if (robot.ready) poseRobot(f);
  (f.mocap || []).forEach((p, k) => loomMeshes[k] && loomMeshes[k].position.fromArray(p));
  if (puffMesh) {
    const pf = f.task && f.task.puffs;
    const m = new THREE.Matrix4();
    puffMesh.count = pf ? Math.min(pf.length, 600) : 0;
    for (let k = 0; k < puffMesh.count; k++) puffMesh.setMatrixAt(k, m.makeTranslation(pf[k][0], pf[k][1], 0.08));
    puffMesh.instanceMatrix.needsUpdate = true;
  }
  if (trail) trail.geometry.setDrawRange(0, i + 1);
  // most recent camera frame at or before i
  for (let k = i; k >= 0 && k > i - 30; k--) {
    if (r.frames[k].fpv) { if (k !== fpvIdx) { $('fpv').src = `data:image/jpeg;base64,${r.frames[k].fpv}`; fpvIdx = k; } break; }
  }
  brainFrame(i);
  updatePathways(f);
  const mode = $('mode');
  mode.textContent = f.mode;
  mode.className = `mode ${f.mode}`;
  $('clock').textContent = `${f.t.toFixed(2)} s`;
  $('scrub').value = i;
}

async function loadReplay(file) {
  if (liveSocket) { liveSocket.close(); liveSocket = null; }
  state.live = false;
  state.playing = false;
  $('play').textContent = 'Play';
  const res = await fetch(`replays/${file}`);
  if (!res.ok) throw new Error(`replay ${file}: HTTP ${res.status}`);
  const r = file.endsWith('.gz')
    ? await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).json()
    : await res.json();
  state.replay = r;
  state.simT = 0;
  lastBrainIdx = -1;
  fpvIdx = -1;
  if (brain.act) brain.act.fill(0);
  buildArena(r);
  $('scrub').max = r.frames.length - 1;
  const hasBrain = r.frames.some((f) => f.brain);
  $('lowlevel').textContent = hasBrain
    ? 'Rotor thrusts, attitude, altitude and wheel speeds come from a conventional flight / drive controller that executes these commands; they are not brain output.'
    : 'Baseline controller: the same sensors and low-level controller, but the decisions come from hand-written rules, not neurons, so the neuron columns stay empty.';
  $('taskline').textContent = `${r.task_title ?? r.task} · ${r.controller} controller · seed ${r.seed}`;
  const m = r.metrics || {};
  const ok = m.success;
  $('outcome').innerHTML = `<b>Outcome:</b> <span class="${ok ? 'ok' : 'fail'}">${ok ? 'task completed' : 'task failed'}</span>${r.outcome_note ? ' &nbsp;' + r.outcome_note : ''}`;
  show(0);
  const f0 = r.frames[0];
  camera.position.set(f0.pos[0] - 1.6, 1.0, -(f0.pos[1]) + 1.2);
  _look.set(f0.pos[0], 0.1, -f0.pos[1]);
  camYaw = null;
  state.playing = true;
  $('play').textContent = 'Pause';
}

// follow camera (positions in three.js coordinates: x, z, -y)
const _v = new THREE.Vector3(), _goal = new THREE.Vector3();
const _look = new THREE.Vector3();
let camYaw = null;
function followCam(dt) {
  const f = state.replay && state.replay.frames[state.idx];
  if (!f) return;
  const [x, y, z] = f.pos;
  const q = f.quat;
  const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  // smooth the heading so fast turns do not whip the camera around
  if (camYaw === null) camYaw = yaw;
  let dy = yaw - camYaw;
  dy = Math.atan2(Math.sin(dy), Math.cos(dy));
  camYaw += dy * (1 - Math.exp(-dt * 2.5));
  _v.set(x, z - 0.05, -y);
  const back = 1.3, up = 0.62;
  _goal.set(x - back * Math.cos(camYaw), z + up, -(y - back * Math.sin(camYaw)));
  const k = 1 - Math.exp(-dt * 6);
  camera.position.lerp(_goal, k);
  _look.lerp(_v, k);
  camera.lookAt(_look);
}

function resize() {
  for (const [rd, cam, el] of [[renderer, camera, canvas], [brenderer, bcam, bcanvas]]) {
    const w = el.clientWidth, h = el.clientHeight;
    if (el.width !== Math.floor(w * rd.getPixelRatio()) || el.height !== Math.floor(h * rd.getPixelRatio())) {
      rd.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
    }
  }
}

function tick(now) {
  const dt = Math.min(0.1, (now - state.last) / 1000);
  state.last = now;
  resize();
  const r = state.replay;
  if (r && state.live) {
    const last = r.frames.length - 1;
    if (last >= 0 && last !== state.idx) show(last);
  } else if (r && state.playing) {
    state.simT += dt * state.speed;
    const i = Math.floor(state.simT / (r.control_dt || 0.02));
    if (i >= r.frames.length - 1) {
      show(r.frames.length - 1);
      state.playing = false;
      $('play').textContent = 'Replay';
    } else if (i !== state.idx) show(i);
  }
  if (state.follow) followCam(dt);
  else {
    const f = r && r.frames[state.idx];
    if (f) controls.target.set(f.pos[0], f.pos[2], -f.pos[1]);
    controls.update();
  }
  if (!reduceMotion) brain.pivot.rotation.y = Math.sin(now / 9000) * 0.45;
  renderer.render(scene, camera);
  brenderer.render(bscene, bcam);
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ UI
$('play').addEventListener('click', () => {
  const r = state.replay;
  if (!r) return;
  if (state.idx >= r.frames.length - 1) { state.simT = 0; lastBrainIdx = -1; show(0); }
  state.playing = !state.playing;
  $('play').textContent = state.playing ? 'Pause' : 'Play';
});
$('scrub').addEventListener('input', (e) => {
  const i = +e.target.value;
  state.simT = i * (state.replay.control_dt || 0.02);
  show(i);
});
$('speed').addEventListener('change', (e) => { state.speed = +e.target.value; });
for (const [id, follow] of [['cam-follow', true], ['cam-free', false]]) {
  $(id).addEventListener('click', () => {
    state.follow = follow;
    controls.enabled = !follow;
    if (!follow) { const f = state.replay && state.replay.frames[state.idx]; if (f) controls.target.set(f.pos[0], f.pos[2], -f.pos[1]); }
    $('cam-follow').classList.toggle('on', follow); $('cam-follow').setAttribute('aria-pressed', follow);
    $('cam-free').classList.toggle('on', !follow); $('cam-free').setAttribute('aria-pressed', !follow);
  });
}
controls.enabled = false;

// ------------------------------------------------------------------ live mode (local server only)
let liveSocket = null;
function startLive(task, controller) {
  if (liveSocket) liveSocket.close();
  const seed = 3000 + Math.floor(Math.random() * 1000);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/live?task=${task}&controller=${controller}&seed=${seed}`);
  liveSocket = ws;
  state.playing = false;
  $('play').textContent = 'Live';
  $('outcome').innerHTML = `<b>Live:</b> starting ${task} with the ${controller} controller (seed ${seed}); loading the brain can take a few seconds…`;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'header') {
      state.replay = { ...msg, frames: [], task_title: `${msg.task} (live)` };
      state.live = true; state.idx = 0; lastBrainIdx = -1; fpvIdx = -1;
      if (brain.act) brain.act.fill(0);
      buildArena(state.replay);
      $('taskline').textContent = `${msg.task} · ${msg.controller} controller · seed ${msg.seed} · live`;
    } else if (msg.type === 'frame' && state.replay) {
      const fr = msg; delete fr.type;
      state.replay.frames.push(fr);
      trailPoint(state.replay.frames.length - 1, fr);
      $('scrub').max = state.replay.frames.length - 1;
    } else if (msg.type === 'done') {
      state.live = false;
      state.replay.metrics = msg.metrics;
      const ok = msg.metrics.success;
      $('outcome').innerHTML = `<b>Live run finished:</b> <span class="${ok ? 'ok' : 'fail'}">${ok ? 'task completed' : 'task failed'}</span>`;
      $('play').textContent = 'Replay';
    } else if (msg.type === 'error') {
      $('outcome').innerHTML = `<span class="fail">Live simulation error: ${msg.message}</span>`;
    }
  };
  ws.onclose = () => { if (state.live) { state.live = false; $('outcome').innerHTML += ' (connection closed)'; } };
}

async function setupLive() {
  try {
    const info = await (await fetch('api/info')).json();
    if (!info.live) return;
    const sel = $('live-task');
    for (const t of info.tasks) { const o = document.createElement('option'); o.value = o.textContent = t; sel.appendChild(o); }
    $('live').hidden = false;
    $('live-run').addEventListener('click', () => startLive(sel.value, $('live-ctrl').value));
  } catch (e) { /* static hosting: no live server */ }
}

function pause() {
  state.playing = false;
  $('play').textContent = 'Play';
}

// small automation hook (used by scripts/capture_viewer.py for README media)
window.connectocopter = {
  frameCount: () => (state.replay ? state.replay.frames.length : 0),
  showFrame: (i) => { pause(); state.simT = i * ((state.replay && state.replay.control_dt) || 0.02); show(i); },
  load: (file) => { $('episode').value = file; return loadReplay(file).then(pause); },
};

async function main() {
  buildPathways();
  await Promise.all([buildRobot(), buildBrain()]);
  const list = await (await fetch('replays/index.json')).json();
  const sel = $('episode');
  for (const e of list) {
    const o = document.createElement('option');
    o.value = e.file;
    o.textContent = e.label;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => loadReplay(sel.value));
  const want = new URLSearchParams(location.search).get('episode');
  if (want && list.some((e) => e.file === want)) sel.value = want;
  setupLive();
  await loadReplay(sel.value);
  const params = new URLSearchParams(location.search);
  if (params.has('t')) {
    const r = state.replay;
    const i = Math.min(r.frames.length - 1, Math.round(+params.get('t') / (r.control_dt || 0.02)));
    state.simT = i * (r.control_dt || 0.02);
    lastBrainIdx = -1;
    show(i);
    if (params.get('paused') === '1') pause();
  }
  if (params.get('cam') === 'free') $('cam-free').click();
  requestAnimationFrame(tick);
}
main().catch((err) => {
  $('outcome').innerHTML = `<span class="fail">Could not load the viewer data (${err.message}). Serve the web/ folder over HTTP, e.g. <code>python -m http.server -d web</code>.</span>`;
  console.error(err);
});
