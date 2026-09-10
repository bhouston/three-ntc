import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { DefaultLoadingManager, MeshStandardMaterial } from "three";
import { WebGPURenderer } from "three/webgpu";
import { NTCLoader, NTCNodeMaterial } from "three-ntc";
import { expect, it } from "vitest";
import { commands } from "vitest/browser";
import { NTCViewer } from "./NTCViewer.js";
import { NTCGridViewer } from "./NTCGridViewer.js";
import brick from "../../public/ntc/brick.ntc?raw";
import hdrUrl from "../../public/textures/equirectangular/san_giuseppe_bridge_2k.hdr?url";

it.each(["single", "trainer", "grid"] as const)(
  "profiles the %s brick viewer canvas after HDR loading",
  async (viewer) => {
    const fixture = document.createElement("div");
    fixture.style.cssText = "width:1024px;height:768px;position:relative";
    document.body.append(fixture);
    const root = createRoot(fixture, {
      onUncaughtError: (error) => {
        errors.push(String(error));
        finish();
      },
    });
    const { cpuModel, channelClassification } = new NTCLoader().parse(brick);
    const material = new NTCNodeMaterial(cpuModel, channelClassification);
    const teacherMaterial = new MeshStandardMaterial({ color: 0x995533 });
    DefaultLoadingManager.setURLModifier((url) => (url.startsWith("/textures/") ? hdrUrl : url));
    const prototype = WebGPURenderer.prototype as any,
      render = prototype.render;
    const frames: { intervalMs: number; completedMs: number }[] = [];
    const errors: string[] = [];
    let renderer: any,
      lastFrame = 0,
      active = true;
    let finish!: () => void;
    const done = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const onError = (event: any) => {
      errors.push(event.error.message);
      finish();
    };
    prototype.render = function (scene: any, camera: any, ...rest: any[]) {
      const start = performance.now();
      const result = render.call(this, scene, camera, ...rest);
      if (active && fixture.contains(this.domElement) && scene.environment && frames.length < 25) {
        if (!renderer) {
          renderer = this;
          renderer.backend.device.addEventListener("uncapturederror", onError);
          renderer.backend.device.lost.then((info: any) => {
            if (active) {
              errors.push(`Device lost: ${info.message}`);
              finish();
            }
          });
        }
        const metric = { intervalMs: lastFrame ? start - lastFrame : 0, completedMs: 0 };
        lastFrame = start;
        frames.push(metric);
        renderer.backend.device.queue
          .onSubmittedWorkDone()
          .then(() => {
            metric.completedMs = performance.now() - start;
            if (frames.length === 25 && frames.every((f) => f.completedMs > 0)) finish();
          })
          .catch((error: unknown) => {
            errors.push(String(error));
            finish();
          });
      }
      return result;
    };
    try {
      root.render(
        viewer === "grid"
          ? createElement(NTCGridViewer, {
              slots: Array.from({ length: 6 }, (_, i) => ({ label: `Brick ${i}`, material })),
            })
          : createElement(NTCViewer, {
              material,
              ...(viewer === "trainer" ? { teacherMaterial, cameraDistance: 2.2 } : {}),
            }),
      );
      await done;
      expect(errors).toEqual([]);
      expect(frames).toHaveLength(25);
      expect(renderer.samples).toBe(0);
      await (commands as any).recordMetric({
        kind: "react-brick-viewer",
        viewer,
        userAgent: navigator.userAgent,
        samples: renderer.samples,
        pixelRatio: window.devicePixelRatio,
        canvas: [renderer.domElement.width, renderer.domElement.height],
        steadyFrames: frames.slice(5),
        errors,
      });
    } finally {
      active = false;
      prototype.render = render;
      renderer?.backend.device.removeEventListener("uncapturederror", onError);
      root.unmount();
      fixture.remove();
      material.dispose();
      teacherMaterial.dispose();
      DefaultLoadingManager.setURLModifier((url) => url);
    }
  },
);
