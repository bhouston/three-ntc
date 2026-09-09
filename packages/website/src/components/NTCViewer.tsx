import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

// Same environment map the three.js NTC examples use, for a matching look.
const HDR_ENVIRONMENT_URL = '/textures/equirectangular/san_giuseppe_bridge_2k.hdr';

export type NTCViewerShape = 'torus' | 'sphere' | 'plane';

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

    renderer.init().then(() => {
      if (disposed) return;
      resize();
      renderer.setAnimationLoop(() => {
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
      renderer.dispose();
      meshRef.current = null;
      teacherMeshRef.current = null;
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
    if (!mesh || !teacherMesh) return;

    if (teacherMaterial) {
      teacherMesh.visible = true;
      teacherMesh.position.x = -1.25;
      mesh.position.x = 1.25;
      teacherMesh.scale.setScalar(0.58);
      mesh.scale.setScalar(0.58);
    } else {
      teacherMesh.visible = false;
      mesh.position.x = 0;
      mesh.scale.setScalar(1);
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
