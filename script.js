/* ==========================================================================
   Manifold Neighborhoods Explorer
   Surface X(u,v) = (u, f(u,v), v), f = A*sin(k1*u)*cos(k2*v). Every ball
   here uses AMBIENT (straight-line, R^3) distance -- no geodesics, no
   tangent plane. Two radii matter:
     - epsilon: a radius the person sets directly.
     - closeness radius: found numerically so the manifold patch inside
       the ball covers a target FRACTION p of the manifold's total surface
       area (or p = (k+1/2)/N, mirroring the Gaussian tool's closeness-ball).
   ========================================================================== */

const DOMAIN = 4.5;
const GRID_RES = 90;
const MAX_SAMPLE_POINTS = 5000;
const SAMPLE_POINT_LIFT = 0.015;
const HIGHLIGHT_POINT_LIFT = 0.022; // drawn after base points, so it sits visibly on top
const REF_POINT_RADIUS = 0.045 * 1.5;
const AREA_GRID = 120;       // resolution of the area-fraction sampling grid

const COLOR_REF_POINT = 0xff4d4d;
const COLOR_POINTS = 0x888c96;
const COLOR_EPS = 0xff8c2b;        // epsilon ball + epsilon-neighborhood
const COLOR_CLOSENESS = 0xff33cc;  // closeness-ball + k-NN

const state = {
  amplitude: 1, kx: 0.5, ky: 0.5,
  u0: 0.7, v0: -0.5,
  eps: 1.0,
  k: 20,
  p: 0.02,
  numSamplePoints: 1000,
  vis: {
    surface: true, points: true, point: true, ambient: false,
    epsBall: false, epsNeighborhood: false, closenessBall: false, knn: false,
    axes: true, grid: false,
  },
};

let needsSurfaceRebuild = true;
let needsIndicatorRebuild = true;

/* ==========================================================================
   Manifold math (no Christoffel symbols / geodesics needed here)
   ========================================================================== */

function f(u, v) {
  const { amplitude: A, kx, ky } = state;
  return A * Math.sin(kx * u) * Math.cos(ky * v);
}
function fu(u, v) {
  const { amplitude: A, kx, ky } = state;
  return A * kx * Math.cos(kx * u) * Math.cos(ky * v);
}
function fv(u, v) {
  const { amplitude: A, kx, ky } = state;
  return -A * ky * Math.sin(kx * u) * Math.sin(ky * v);
}
function point3D(u, v) {
  return new THREE.Vector3(u, f(u, v), v);
}
// First fundamental form, needed only for the surface-area element below.
function metricAt(u, v) {
  const Fu = fu(u, v), Fv = fv(u, v);
  return { E: 1 + Fu * Fu, F: Fu * Fv, G: 1 + Fv * Fv };
}

/* ==========================================================================
   Area-fraction engine for the closeness-ball. A grid of cells covers the
   rendered domain; each cell's area contribution is dA = sqrt(EG-F^2) * du
   * dv (the standard surface-area element). Cells are sorted once by
   ambient distance to c, giving a cumulative-area array that converts
   between radius and area-fraction in O(log n), without re-integrating on
   every query.
   ========================================================================== */

let areaCache = null;

function rebuildAreaCache() {
  const n = AREA_GRID;
  const cellU = (2 * DOMAIN) / n, cellV = (2 * DOMAIN) / n;
  const P = point3D(state.u0, state.v0);
  const count = n * n;
  const dists = new Float64Array(count);
  const areas = new Float64Array(count);
  let idx = 0;
  for (let i = 0; i < n; i++) {
    const v = -DOMAIN + (i + 0.5) * cellV;
    for (let j = 0; j < n; j++) {
      const u = -DOMAIN + (j + 0.5) * cellU;
      const { E, F, G } = metricAt(u, v);
      areas[idx] = Math.sqrt(Math.max(0, E * G - F * F)) * cellU * cellV;
      const dx = u - P.x, dy = f(u, v) - P.y, dz = v - P.z;
      dists[idx] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      idx++;
    }
  }
  const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => dists[a] - dists[b]);
  const sortedDist = new Float64Array(count);
  const cumArea = new Float64Array(count);
  let cum = 0;
  for (let i = 0; i < count; i++) {
    cum += areas[order[i]];
    sortedDist[i] = dists[order[i]];
    cumArea[i] = cum;
  }
  areaCache = { sortedDist, cumArea, totalArea: cum };
}

function areaFractionAtRadius(r) {
  if (!areaCache) return 0;
  const { sortedDist, cumArea, totalArea } = areaCache;
  let lo = 0, hi = sortedDist.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sortedDist[mid] <= r) { idx = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return idx >= 0 ? cumArea[idx] / totalArea : 0;
}

function radiusForAreaFraction(p) {
  if (!areaCache) return 0;
  const { sortedDist, cumArea, totalArea } = areaCache;
  const target = Math.min(Math.max(p, 0), 1) * totalArea;
  let lo = 0, hi = sortedDist.length - 1, idx = sortedDist.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cumArea[mid] >= target) { idx = mid; hi = mid - 1; } else lo = mid + 1;
  }
  return sortedDist[idx];
}

/* ==========================================================================
   Three.js scene setup
   ========================================================================== */

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161c);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 100);
const DEFAULT_CAM_POS = new THREE.Vector3(5.5, 4.5, 6.5);
camera.position.copy(DEFAULT_CAM_POS);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(6, 10, 4);
scene.add(dirLight);
const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.35);
dirLight2.position.set(-6, -4, -6);
scene.add(dirLight2);

const gridHelper = new THREE.GridHelper(10, 20, 0x2b2f3a, 0x22252e);
scene.add(gridHelper);

const axesHelper = new THREE.AxesHelper(3.0);
scene.add(axesHelper);
function updateAxesPlacement() {
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
  const signX = dir.x >= 0 ? -1 : 1;
  const signZ = dir.z >= 0 ? -1 : 1;
  axesHelper.position.set(
    signX * DOMAIN * 0.55,
    -Math.max(1.0, Math.abs(state.amplitude) + 0.6),
    signZ * DOMAIN * 0.55
  );
}

/* ---- Surface mesh ---- */
let surfaceGeom = new THREE.BufferGeometry();
const surfaceMat = new THREE.MeshStandardMaterial({
  color: 0x8892a0, side: THREE.DoubleSide, metalness: 0.05, roughness: 0.85,
  transparent: true, opacity: 0.92,
});
const surfaceMesh = new THREE.Mesh(surfaceGeom, surfaceMat);
scene.add(surfaceMesh);

function rebuildSurface() {
  const n = GRID_RES;
  const positions = new Float32Array((n + 1) * (n + 1) * 3);
  let idx = 0;
  for (let i = 0; i <= n; i++) {
    const v = -DOMAIN + (2 * DOMAIN * i) / n;
    for (let j = 0; j <= n; j++) {
      const u = -DOMAIN + (2 * DOMAIN * j) / n;
      positions[idx++] = u;
      positions[idx++] = f(u, v);
      positions[idx++] = v;
    }
  }
  const indices = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = i * (n + 1) + j, b = a + 1, c = a + (n + 1), d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  surfaceGeom.dispose();
  surfaceGeom = new THREE.BufferGeometry();
  surfaceGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  surfaceGeom.setIndex(indices);
  surfaceGeom.computeVertexNormals();
  surfaceMesh.geometry = surfaceGeom;
  regionMesh.geometry = surfaceGeom; // the ball overlay shares the manifold's mesh
}

/* ---- Sample points: base grey layer (cached (u,v), recomputed heights on
   shape change) plus two highlight layers (eps-neighborhood, k-NN) that
   are rebuilt from scratch each time c/eps/k/N changes -- small enough to
   be cheap, and simpler than filtering the base buffer. ---- */
function makeCircleTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}
const circleTex = makeCircleTexture();

const sampleUV = (() => {
  const arr = [];
  for (let i = 0; i < MAX_SAMPLE_POINTS; i++) {
    arr.push({ u: (Math.random() * 2 - 1) * DOMAIN, v: (Math.random() * 2 - 1) * DOMAIN });
  }
  return arr;
})();
let samplePos = new Array(MAX_SAMPLE_POINTS); // THREE.Vector3, recomputed on shape change

let samplePointsGeom = new THREE.BufferGeometry();
const samplePointsMat = new THREE.PointsMaterial({
  color: COLOR_POINTS, size: REF_POINT_RADIUS * 2, sizeAttenuation: true,
  map: circleTex, alphaTest: 0.5,
});
const samplePointsObj = new THREE.Points(samplePointsGeom, samplePointsMat);
scene.add(samplePointsObj);

function rebuildSamplePoints() {
  const positions = new Float32Array(MAX_SAMPLE_POINTS * 3);
  for (let i = 0; i < MAX_SAMPLE_POINTS; i++) {
    const { u, v } = sampleUV[i];
    const y = f(u, v);
    samplePos[i] = new THREE.Vector3(u, y, v);
    positions[i * 3] = u;
    positions[i * 3 + 1] = y + SAMPLE_POINT_LIFT;
    positions[i * 3 + 2] = v;
  }
  samplePointsGeom.dispose();
  samplePointsGeom = new THREE.BufferGeometry();
  samplePointsGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  samplePointsGeom.setDrawRange(0, state.numSamplePoints);
  samplePointsObj.geometry = samplePointsGeom;
}

function makeHighlightPoints(color) {
  const geom = new THREE.BufferGeometry();
  const mat = new THREE.PointsMaterial({
    color, size: REF_POINT_RADIUS * 2.3, sizeAttenuation: true,
    map: circleTex, alphaTest: 0.5,
  });
  const obj = new THREE.Points(geom, mat);
  scene.add(obj);
  return obj;
}
const epsNeighborhoodObj = makeHighlightPoints(COLOR_EPS);
const knnObj = makeHighlightPoints(COLOR_CLOSENESS);

function setHighlightPoints(obj, indices) {
  const positions = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const p = samplePos[indices[i]];
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y + HIGHLIGHT_POINT_LIFT;
    positions[i * 3 + 2] = p.z;
  }
  obj.geometry.dispose();
  obj.geometry = new THREE.BufferGeometry();
  obj.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
}

/* ---- Reference point c ---- */
const pointMesh = new THREE.Mesh(
  new THREE.SphereGeometry(REF_POINT_RADIUS, 20, 20),
  new THREE.MeshStandardMaterial({ color: COLOR_REF_POINT, emissive: 0x551515 })
);
scene.add(pointMesh);

/* ---- Ambient sphere (faint, radius = epsilon) ---- */
const ambientGeom = new THREE.SphereGeometry(1, 32, 24);
const ambientMat = new THREE.MeshBasicMaterial({
  color: COLOR_EPS, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false,
});
const ambientMesh = new THREE.Mesh(ambientGeom, ambientMat);
scene.add(ambientMesh);

/* ---- Epsilon ball & closeness-ball: border only, drawn per pixel ----
   Each ball is shown as just its boundary line on the manifold (no fill).
   An overlay shares the manifold's own mesh and, for every pixel, checks
   how close the straight-line 3D distance to c is to the ball's radius;
   pixels within a hair of it get the ball's color. Because the overlay IS
   the surface, the line is clipped exactly at the manifold's border, and its
   on-screen width stays constant (via fwidth) however the surface is tilted. */
const regionMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  extensions: { derivatives: true },
  uniforms: {
    uC: { value: new THREE.Vector3() },
    uREps: { value: 1 },
    uRClose: { value: 1 },
    uShowE: { value: 0 },
    uShowC: { value: 0 },
    uColE: { value: new THREE.Color(COLOR_EPS) },
    uColC: { value: new THREE.Color(COLOR_CLOSENESS) },
  },
  vertexShader: `
    varying vec3 vPos;
    void main() {
      vPos = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    precision highp float;
    uniform vec3 uC;
    uniform float uREps, uRClose, uShowE, uShowC;
    uniform vec3 uColE, uColC;
    varying vec3 vPos;
    void main() {
      float d = length(vPos - uC);
      float w = clamp(fwidth(d) * 1.5, 0.0005, 0.08); // ~constant pixel width
      float aE = uShowE > 0.5 ? 1.0 - smoothstep(w * 0.5, w, abs(d - uREps)) : 0.0;
      float aC = uShowC > 0.5 ? 1.0 - smoothstep(w * 0.5, w, abs(d - uRClose)) : 0.0;
      float a = max(aE, aC);
      if (a < 0.02) discard;
      vec3 col = aC > aE ? uColC : uColE;
      gl_FragColor = vec4(col, a);
    }
  `,
});
const regionMesh = new THREE.Mesh(surfaceGeom, regionMat);
regionMesh.renderOrder = 1;
scene.add(regionMesh);


/* ==========================================================================
   Rebuild indicators
   ========================================================================== */

let closenessRadius = 0;

function rebuildIndicators() {
  const { u0, v0 } = state;
  const P = point3D(u0, v0);
  pointMesh.position.copy(P);

  ambientMesh.position.copy(P);
  ambientMesh.scale.setScalar(state.eps);

  rebuildAreaCache();
  closenessRadius = radiusForAreaFraction(state.p);

  // Epsilon ball and closeness-ball are painted per pixel by regionMat.
  regionMat.uniforms.uC.value.copy(P);
  regionMat.uniforms.uREps.value = state.eps;
  regionMat.uniforms.uRClose.value = closenessRadius;

  // k-NN and epsilon-neighborhood among the currently active sample points
  const activeN = state.numSamplePoints;
  const withDist = [];
  for (let i = 0; i < activeN; i++) withDist.push({ i, d: samplePos[i].distanceTo(P) });
  withDist.sort((a, b) => a.d - b.d);
  const kClamped = Math.min(state.k, withDist.length);
  setHighlightPoints(knnObj, withDist.slice(0, kClamped).map((o) => o.i));
  const epsIndices = withDist.filter((o) => o.d <= state.eps).map((o) => o.i);
  setHighlightPoints(epsNeighborhoodObj, epsIndices);

  const knnDist = withDist.length ? withDist[kClamped - 1].d : NaN;
  const N = Math.max(1, state.numSamplePoints);
  const targetKN = (state.k + 0.5) / N;
  document.getElementById('readout').textContent =
    `c = (${u0.toFixed(2)}, ${f(u0, v0).toFixed(2)}, ${v0.toFixed(2)})\n` +
    `ε ball area fraction ≈ ${areaFractionAtRadius(state.eps).toFixed(4)}\n` +
    `k-th NN distance ≈ ${knnDist.toFixed(3)}\n` +
    `(k+½)/N = ${targetKN.toFixed(4)}\n` +
    `closeness target p = ${state.p.toFixed(4)}\n` +
    `closeness radius r* ≈ ${closenessRadius.toFixed(3)}\n` +
    `achieved area fraction ≈ ${areaFractionAtRadius(closenessRadius).toFixed(4)}`;
}

function applyVisibility() {
  const v = state.vis;
  surfaceMesh.visible = v.surface;
  samplePointsObj.visible = v.points;
  pointMesh.visible = v.point;
  ambientMesh.visible = v.ambient;
  regionMat.uniforms.uShowE.value = v.epsBall ? 1 : 0;
  regionMesh.visible = v.surface && (v.epsBall || v.closenessBall);
  epsNeighborhoodObj.visible = v.epsNeighborhood;
  regionMat.uniforms.uShowC.value = v.closenessBall ? 1 : 0;
  knnObj.visible = v.knn;
  axesHelper.visible = v.axes;
  gridHelper.visible = v.grid;
}

/* ==========================================================================
   UI wiring
   ========================================================================== */

function bindSlider(id, stateKey, decimals, onChange) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + '-val');
  const update = () => {
    const val = parseFloat(input.value);
    state[stateKey] = val;
    out.textContent = val.toFixed(decimals);
    onChange();
  };
  input.addEventListener('input', update);
  update();
}

bindSlider('u0', 'u0', 2, () => { needsIndicatorRebuild = true; });
bindSlider('v0', 'v0', 2, () => { needsIndicatorRebuild = true; });
bindSlider('amp', 'amplitude', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('kx', 'kx', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('ky', 'ky', 2, () => { needsSurfaceRebuild = true; needsIndicatorRebuild = true; });
bindSlider('eps', 'eps', 2, () => { needsIndicatorRebuild = true; });
bindSlider('k', 'k', 0, () => { needsIndicatorRebuild = true; });
bindSlider('p', 'p', 3, () => { needsIndicatorRebuild = true; });
bindSlider('numPoints', 'numSamplePoints', 0, () => { needsIndicatorRebuild = true; });

document.getElementById('fit-p').addEventListener('click', () => {
  const N = Math.max(1, state.numSamplePoints);
  const target = Math.min(1, (state.k + 0.5) / N);
  state.p = target;
  const pInput = document.getElementById('p');
  pInput.value = target;
  document.getElementById('p-val').textContent = target.toFixed(3);
  needsIndicatorRebuild = true;
});

document.getElementById('reset-point').addEventListener('click', () => {
  state.u0 = 0; state.v0 = 0;
  document.getElementById('u0').value = 0;
  document.getElementById('v0').value = 0;
  document.getElementById('u0-val').textContent = '0.00';
  document.getElementById('v0-val').textContent = '0.00';
  needsIndicatorRebuild = true;
});

const visMap = {
  'vis-surface': 'surface', 'vis-points': 'points', 'vis-point': 'point', 'vis-ambient': 'ambient',
  'vis-eps': 'epsBall', 'vis-eps-neigh': 'epsNeighborhood',
  'vis-closeness': 'closenessBall', 'vis-knn': 'knn',
  'vis-axes': 'axes', 'vis-grid': 'grid',
};
Object.entries(visMap).forEach(([id, key]) => {
  document.getElementById(id).addEventListener('change', (e) => {
    state.vis[key] = e.target.checked;
    applyVisibility();
  });
});

const panel = document.getElementById('panel');
document.getElementById('panel-toggle').addEventListener('click', () => {
  panel.classList.toggle('open');
});

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(surfaceMesh, false)[0];
  if (!hit) return;
  const u = THREE.MathUtils.clamp(hit.point.x, -DOMAIN, DOMAIN);
  const v = THREE.MathUtils.clamp(hit.point.z, -DOMAIN, DOMAIN);
  state.u0 = u; state.v0 = v;
  document.getElementById('u0').value = u;
  document.getElementById('v0').value = v;
  document.getElementById('u0-val').textContent = u.toFixed(2);
  document.getElementById('v0-val').textContent = v.toFixed(2);
  needsIndicatorRebuild = true;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ==========================================================================
   Photo / video capture
   ========================================================================== */

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

document.getElementById('capture-photo').addEventListener('click', () => {
  renderer.render(scene, camera);
  renderer.domElement.toBlob((blob) => {
    if (blob) downloadBlob(blob, `manifold-balls-${Date.now()}.png`);
  }, 'image/png');
});

const videoBtn = document.getElementById('capture-video');
const canCaptureVideo = !!(renderer.domElement.captureStream && window.MediaRecorder);
if (!canCaptureVideo) {
  videoBtn.disabled = true;
  videoBtn.title = 'Video recording is not supported in this browser (works in Chrome/Firefox/Edge; limited on iOS Safari).';
  videoBtn.textContent = '⏺ Record Video (unsupported)';
} else {
  let mediaRecorder = null;
  let recordedChunks = [];
  videoBtn.addEventListener('click', () => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      const stream = renderer.domElement.captureStream(30);
      let mimeType = 'video/webm;codecs=vp9';
      if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
      try {
        mediaRecorder = new MediaRecorder(stream, { mimeType });
      } catch (err) {
        alert('Could not start video recording: ' + err.message);
        return;
      }
      recordedChunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        downloadBlob(blob, `manifold-balls-${Date.now()}.webm`);
        videoBtn.textContent = '⏺ Record Video';
        videoBtn.classList.remove('recording');
      };
      mediaRecorder.start();
      videoBtn.textContent = '⏹ Stop Recording';
      videoBtn.classList.add('recording');
    } else {
      mediaRecorder.stop();
    }
  });
}

/* ==========================================================================
   Main loop
   ========================================================================== */

applyVisibility();

function animate() {
  requestAnimationFrame(animate);
  if (needsSurfaceRebuild) { rebuildSurface(); rebuildSamplePoints(); needsSurfaceRebuild = false; needsIndicatorRebuild = true; }
  if (needsIndicatorRebuild) { rebuildIndicators(); needsIndicatorRebuild = false; }
  updateAxesPlacement();
  controls.update();
  renderer.render(scene, camera);
}
animate();
