import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

// Same environment map the three.js NTC examples use, for a matching look.
const HDR_ENVIRONMENT_URL = '/textures/equirectangular/san_giuseppe_bridge_2k.hdr';

// Slow, constant spin so viewers can tell the mesh is 3D: one full turn every 30s.
const ROTATION_SPEED = (2 * Math.PI) / 30;

export type NTCViewerShape = 'torus' | 'sphere' | 'plane';

// A text label rendered onto a plane instead of an HTML overlay, so it lives
// in the scene and pans/rotates/zooms with the meshes. Faces +z (the
// camera's start orientation) and is never re-oriented after that - per
// product decision, it doesn't need to billboard as the user orbits.
function makeTextLabel(text: string): any {
  const fontSize = 64;
  const canvas = document.createElement('canvas');
  const measureCtx = canvas.getContext('2d')!;
  measureCtx.font = `${fontSize}px sans-serif`;
  const width = Math.ceil(measureCtx.measureText(text).width) + 40;
  const height = fontSize + 40;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, width / 2, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const planeHeight = 0.16;
  const geometry = new THREE.PlaneGeometry((planeHeight * width) / height, planeHeight);
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 999;
  mesh.visible = false;
  return mesh;
}

function buildShapeGeometry(shape: NTCViewerShape): any {
  const geometry =
    shape === 'torus'
      ? new THREE.TorusGeometry(0.7, 0.28, 32, 96)
      : shape === 'plane'
        ? new THREE.PlaneGeometry(1.6, 1.6)
        : new THREE.SphereGeometry(1, 64, 64);

  // Normal/clearcoatNormal channels decode in tangent space (see
  // NTCNodeMaterial.reconstructFinalNormal) - every shape needs a real
  // tangentWorld/bitangentWorld basis to blend through.
  geometry.computeTangents();
  return geometry;
}

/**
 * Owns the WebGPU canvas: renderer, camera, orbit controls, and either one
 * centered mesh (`material` alone) or two side-by-side meshes sharing one
 * scene/camera - `teacherMaterial` on the left (the MaterialX source,
 * unmodified by NTC), `material` (the live NTCNodeMaterial preview) on the
 * right - matching the original three.js example's teacherMesh/neuralMesh
 * pair. Swaps materials/geometry in place rather than tearing down the
 * renderer on every change.
 */
export function NTCViewer({
  material,
  teacherMaterial = null,
  shape = 'sphere',
}: {
  material: any;
  teacherMaterial?: any;
  shape?: NTCViewerShape;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const meshRef = useRef<any>(null);
  const teacherMeshRef = useRef<any>(null);
  const labelRef = useRef<any>(null);
  const teacherLabelRef = useRef<any>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 3);

    const renderer = new WebGPURenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.5);
    key.position.set(3, 4, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1);
    fill.position.set(-4, -2, -3);
    scene.add(fill);

    const placeholderMaterial = new THREE.MeshStandardMaterial({ color: 0x888888 });

    const mesh = new THREE.Mesh(buildShapeGeometry(shape), placeholderMaterial);
    scene.add(mesh);
    meshRef.current = mesh;

    // Shares the same placeholder material instance until a real
    // `teacherMaterial` is supplied - never disposed by this component (see
    // the teacherMaterial-swap effect below), since its lifecycle belongs to
    // whoever loaded it (e.g. a MaterialXLoader result kept alive elsewhere).
    const teacherMesh = new THREE.Mesh(mesh.geometry, placeholderMaterial);
    teacherMesh.visible = false;
    scene.add(teacherMesh);
    teacherMeshRef.current = teacherMesh;

    const label = makeTextLabel('NTC material');
    scene.add(label);
    labelRef.current = label;

    const teacherLabel = makeTextLabel('MaterialX teacher');
    scene.add(teacherLabel);
    teacherLabelRef.current = teacherLabel;

    let disposed = false;
    let envTexture: any = null;

    new HDRLoader().load(HDR_ENVIRONMENT_URL, (texture: any) => {
      if (disposed) return;
      texture.mapping = THREE.EquirectangularReflectionMapping;
      envTexture = texture;
      scene.environment = texture;
      scene.background = texture;
    });

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const { clientWidth: width, clientHeight: height } = parent;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };

    const ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    const clock = new THREE.Clock();

    renderer.init().then(() => {
      if (disposed) return;
      resize();
      renderer.setAnimationLoop(() => {
        const dt = clock.getDelta();
        mesh.rotation.y += ROTATION_SPEED * dt;
        teacherMesh.rotation.y += ROTATION_SPEED * dt;
        controls.update();
        renderer.render(scene, camera);
      });
    });

    return () => {
      disposed = true;
      ro.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      mesh.geometry.dispose();
      mesh.material?.dispose && mesh.material.dispose();
      envTexture?.dispose();
      label.geometry.dispose();
      label.material.map.dispose();
      label.material.dispose();
      teacherLabel.geometry.dispose();
      teacherLabel.material.map.dispose();
      teacherLabel.material.dispose();
      renderer.dispose();
      meshRef.current = null;
      teacherMeshRef.current = null;
      labelRef.current = null;
      teacherLabelRef.current = null;
    };
  }, []);

  // Swaps the neural preview material - disposes the previous one, since
  // this component (via rebuildPreviewMaterial in the trainer page) owns its
  // lifecycle.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || !material) return;
    const previous = mesh.material;
    mesh.material = material;
    if (previous?.dispose && previous !== material && previous !== teacherMeshRef.current?.material) previous.dispose();
  }, [material]);

  // Swaps the teacher-side material - never disposed here (see the
  // component doc comment above).
  useEffect(() => {
    const teacherMesh = teacherMeshRef.current;
    if (!teacherMesh || !teacherMaterial) return;
    teacherMesh.material = teacherMaterial;
  }, [teacherMaterial]);

  // Toggles side-by-side layout: two smaller, offset meshes when a teacher
  // material is present, one centered mesh otherwise.
  useEffect(() => {
    const mesh = meshRef.current;
    const teacherMesh = teacherMeshRef.current;
    const label = labelRef.current;
    const teacherLabel = teacherLabelRef.current;
    if (!mesh || !teacherMesh || !label || !teacherLabel) return;

    if (teacherMaterial) {
      teacherMesh.visible = true;
      teacherMesh.position.x = -0.625;
      mesh.position.x = 0.625;
      teacherMesh.scale.setScalar(0.58);
      mesh.scale.setScalar(0.58);

      label.visible = true;
      teacherLabel.visible = true;
      label.position.set(mesh.position.x, 0.75, 0);
      teacherLabel.position.set(teacherMesh.position.x, 0.75, 0);
    } else {
      teacherMesh.visible = false;
      mesh.position.x = 0;
      mesh.scale.setScalar(1);

      label.visible = false;
      teacherLabel.visible = false;
    }
  }, [Boolean(teacherMaterial)]);

  useEffect(() => {
    const mesh = meshRef.current;
    const teacherMesh = teacherMeshRef.current;
    if (!mesh) return;
    const previous = mesh.geometry;
    const geometry = buildShapeGeometry(shape);
    mesh.geometry = geometry;
    if (teacherMesh) teacherMesh.geometry = geometry;
    previous?.dispose();
  }, [shape]);

  return <canvas ref={canvasRef} className="block h-full w-full" />;
}
