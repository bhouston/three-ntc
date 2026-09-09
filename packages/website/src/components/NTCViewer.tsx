import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Owns the WebGPU canvas: renderer, camera, orbit controls, a single sphere
 * mesh and a render loop. Swaps `mesh.material` whenever `material` changes.
 */
export function NTCViewer({ material }: { material: any }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const meshRef = useRef<any>(null);

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

    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), new THREE.MeshStandardMaterial({ color: 0x888888 }));
    scene.add(mesh);
    meshRef.current = mesh;

    let disposed = false;

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
      renderer.dispose();
      meshRef.current = null;
    };
  }, []);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || !material) return;
    const previous = mesh.material;
    mesh.material = material;
    if (previous?.dispose && previous !== material) previous.dispose();
  }, [material]);

  return <canvas ref={canvasRef} className="block h-full w-full" />;
}
