---
title: Getting started
sidebar_position: 1
description: Load a neural material into a three.js WebGPU scene.
---

three-ntc stores a material's varying texture channels in a learned feature grid and a small neural decoder. The runtime reconstructs those channels inside a three.js node material. Train once, export a `.ntc` asset, and load that asset in your application without shipping the trainer.

The project has two published packages:

| Package | Purpose |
| --- | --- |
| `three-ntc` | Load `.ntc` assets and render `NTCNodeMaterial`. |
| `three-ntc-trainer` | Bake materials, train models with GPU compute, and export `.ntc` assets. |

## Install

```sh
npm install three three-ntc
npm install --save-dev @types/three
```

The runtime declares `three >= 0.180.0` as a peer dependency. The repository currently develops against three.js 0.186.0; keep your three.js runtime and types aligned. Use a bundler that understands the `three/webgpu` and `three/tsl` entry points. The package declares Node.js 22 or newer for tooling.

## Browser requirements

Use a browser and physical GPU with WebGPU support, with hardware acceleration enabled. WebGPU requires a secure context: serve the application over HTTPS, or use localhost during development. Training additionally needs GPU compute and enough device resources for the selected model and batch size. Browser support alone does not guarantee that a particular GPU can compile or run a large model.

The documentation is static HTML; the interactive viewer and trainer run in the browser. GPU work cannot run during server-side rendering. Initialize renderers inside your framework's client lifecycle and release them when the view unmounts.

## Render a model

This complete example expects a browser bundler and a `gold.ntc` asset served from your application's public directory. Use the viewer's bundled models for experimentation, or export one with the [training guide](./training.md).

```ts
import * as THREE from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

export async function mountNTCDemo(container: HTMLElement) {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');

  const renderer = new WebGPURenderer({ antialias: true });
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  try {
    const { cpuModel, channelClassification } =
      await new NTCLoader().loadAsync('/gold.ntc');
    const material = new NTCNodeMaterial(cpuModel, channelClassification);
    const geometry = new THREE.SphereGeometry(1, 48, 32);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x17202a);
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 3));
    const light = new THREE.DirectionalLight(0xffffff, 4);
    light.position.set(3, 4, 5);
    scene.add(light);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 3.5;
    const resize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener('resize', resize);
    renderer.setAnimationLoop(() => {
      mesh.rotation.y += 0.003;
      renderer.render(scene, camera);
    });

    return () => {
      renderer.setAnimationLoop(null);
      window.removeEventListener('resize', resize);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  } catch (error) {
    renderer.dispose();
    renderer.domElement.remove();
    throw error;
  }
}
```

Give the container an explicit height, for example `height: 500px`. Call the returned cleanup function when leaving the page. For polished metallic materials, add an appropriate environment map and dispose it when your application no longer uses it.

## Next steps

Read [runtime integration](./runtime.md) for filtering, UV transforms, debugging, and asset handling. Read [training and export](./training.md) to create assets. The [theory explainer](./theory.md) describes how the representation works and the tradeoffs behind it.
