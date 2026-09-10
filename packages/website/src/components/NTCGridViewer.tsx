import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

// Same environment map NTCViewer uses, for a matching look.
const HDR_ENVIRONMENT_URL = '/textures/equirectangular/san_giuseppe_bridge_2k.hdr';

const COLUMNS = 3;
const SPACING = 2.4;

// Slow, constant spin so viewers can tell each mesh is 3D: one full turn every 30s.
const ROTATION_SPEED = (2 * Math.PI) / 30;

export interface NTCGridSlot {
  label: string;
  material: any | null;
}

/**
 * One shared WebGPU scene/camera/OrbitControls showing every slot's material
 * on its own sphere laid out in a 3-column grid - rotate/zoom/pan moves the
 * whole grid together, instead of NTCViewer's per-material renderer.
 */
export function NTCGridViewer({ slots }: { slots: NTCGridSlot[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const meshesRef = useRef<any[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const rows = Math.ceil(slots.length / COLUMNS);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, SPACING * Math.max(COLUMNS, rows) * 1.3);

    // Match NTCViewer: MSAA causes severe WebKit slowdowns at Retina resolutions.
    const renderer = new WebGPURenderer({ canvas, antialias: false });
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

    const geometry = new THREE.SphereGeometry(1, 64, 64);
    geometry.computeTangents();

    const placeholderMaterial = new THREE.MeshStandardMaterial({ color: 0x888888 });

    const originX = ((Math.min(COLUMNS, slots.length) - 1) * SPACING) / 2;
    const originY = ((rows - 1) * SPACING) / 2;

    const meshes = slots.map((slot, i) => {
      const mesh = new THREE.Mesh(geometry, placeholderMaterial);
      mesh.position.set((i % COLUMNS) * SPACING - originX, originY - Math.floor(i / COLUMNS) * SPACING, 0);
      mesh.scale.setScalar(0.85);
      mesh.visible = Boolean(slot.material);
      scene.add(mesh);
      return mesh;
    });
    meshesRef.current = meshes;

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
        for (const mesh of meshes) mesh.rotation.y += ROTATION_SPEED * dt;
        controls.update();
        renderer.render(scene, camera);
      });
    });

    return () => {
      disposed = true;
      ro.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      geometry.dispose();
      placeholderMaterial.dispose();
      envTexture?.dispose();
      renderer.dispose();
      meshesRef.current = [];
    };
    // Grid structure (slot count/labels) is fixed for the lifetime of this
    // component - only the materials inside `slots` change, handled below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slots.length]);

  // Swaps each slot's material in place as it finishes loading.
  useEffect(() => {
    slots.forEach((slot, i) => {
      const mesh = meshesRef.current[i];
      if (!mesh || !slot.material || mesh.material === slot.material) return;
      mesh.material = slot.material;
      mesh.visible = true;
    });
  }, [slots]);

  return <canvas ref={canvasRef} className="block h-full w-full" />;
}
