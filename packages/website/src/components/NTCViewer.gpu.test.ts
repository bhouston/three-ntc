import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { DefaultLoadingManager, MeshStandardMaterial } from "three";
import { WebGPURenderer } from "three/webgpu";
import { NTCLoader, NTCNodeMaterial, type NTCSamplingMode } from "three-ntc";
import { expect, it } from "vitest";
import { commands } from "vitest/browser";
import { summarizeTimings } from "../../../../test/performance-metrics.js";
import { NTCViewer } from "./NTCViewer.js";
import { NTCGridViewer } from "./NTCGridViewer.js";
import brick from "../../public/ntc/brick.ntc?raw";
import hdrUrl from "../../public/textures/equirectangular/san_giuseppe_bridge_2k.hdr?url";

// Long enough to expose queue buildup after the initial pipeline compilation.
const frameCount = 120;
const warmupFrames = 20;
const cases = (["single", "trainer", "grid"] as const).flatMap((viewer) =>
  [512, 1024].map((width) => ({ viewer, width, samplingMode: "nearest" as NTCSamplingMode })),
);
cases.push(
  {viewer: 'single', width: 1024, samplingMode: 'stochastic'},
  {viewer: 'single', width: 1024, samplingMode: 'trilinear'},
);
it.each(cases)(
  "profiles the $viewer brick viewer ($samplingMode) at $width CSS pixels after HDR loading",
  async ({ viewer, width, samplingMode }) => {
    const fixture = document.createElement("div");
    fixture.style.cssText = `width:${width}px;height:${width * 0.75}px;position:relative`;
    document.body.append(fixture);
    const root = createRoot(fixture, {
      onUncaughtError: (error) => {
        errors.push(String(error));
        finish();
      },
    });
    const { cpuModel, channelClassification } = new NTCLoader().parse(brick);
    const material = new NTCNodeMaterial(cpuModel, channelClassification, {samplingMode});
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
    const mountedAt = performance.now();
    let firstFrameMs = 0;
    const heartbeatGaps: number[] = [];
    let lastHeartbeat = mountedAt;
    const heartbeat = setInterval(() => {
      const now = performance.now();
      heartbeatGaps.push(now - lastHeartbeat);
      lastHeartbeat = now;
    }, 16);
    const onError = (event: any) => {
      errors.push(event.error.message);
      finish();
    };
    prototype.render = function (scene: any, camera: any, ...rest: any[]) {
      const start = performance.now();
      const result = render.call(this, scene, camera, ...rest);
      if (
        active &&
        fixture.contains(this.domElement) &&
        scene.environment &&
        frames.length < frameCount
      ) {
        if (!renderer) {
          firstFrameMs = start - mountedAt;
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
            if (frames.length === frameCount && frames.every((f) => f.completedMs > 0)) finish();
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
      expect(frames).toHaveLength(frameCount);
      expect(renderer.samples).toBe(0);
      expect(heartbeatGaps.length).toBeGreaterThan(0);
      const steadyFrames = frames.slice(warmupFrames);
      await (commands as any).recordMetric({
        kind: "react-brick-viewer",
        viewer,
        samplingMode,
        cssSize: [width, width * 0.75],
        warmupFrames,
        firstFrameMs, // Includes HDR loading and initial pipeline work.
        adapter: renderer.backend.device.adapterInfo?.description ?? "unavailable",
        userAgent: navigator.userAgent,
        samples: renderer.samples,
        pixelRatio: window.devicePixelRatio,
        canvas: [renderer.domElement.width, renderer.domElement.height],
        steadyFrames,
        summary: {
          interval: summarizeTimings(steadyFrames.map((frame) => frame.intervalMs)),
          completion: summarizeTimings(steadyFrames.map((frame) => frame.completedMs)),
          heartbeat: summarizeTimings(heartbeatGaps),
        },
        heartbeatGaps,
        errors,
      });
    } finally {
      active = false;
      clearInterval(heartbeat);
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
