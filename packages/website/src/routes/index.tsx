import { createFileRoute, Link } from '@tanstack/react-router';
import { useEffect, useState } from 'react';

import { NTCGridViewer, type NTCGridSlot } from '@/components/NTCGridViewer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EXAMPLE_FILES, parseNtc } from '@/lib/ntc-examples';
import { seoMeta } from '@/lib/seo';

export const Route = createFileRoute('/')({
  component: HomePage,
  head: () => ({
    meta: seoMeta({
      title: 'Three-NTC: Neural Texture Compression for Three.js',
      description:
        "A three.js implementation of NVIDIA's Neural Texture Compression paper — materials as a tiny neural network instead of a stack of full-resolution textures.",
      path: '/',
    }),
  }),
});

const NTC_PAPER_URL = 'https://research.nvidia.com/labs/rtr/neural_texture_compression/';

// 3 columns, EXAMPLE_FILES has 8 entries: two full rows of 3, then a
// centered row of 2.

function HomePage() {
  const [slots, setSlots] = useState<NTCGridSlot[]>(() =>
    EXAMPLE_FILES.map(() => ({ label: '', material: null })),
  );

  useEffect(() => {
    let cancelled = false;

    // Fetch every file's text in parallel (cheap network round-trips), but
    // parse/build each material in series so they appear one at a time
    // instead of all popping in together - gets the first material on
    // screen sooner.
    void (async () => {
      const texts = await Promise.all(
        EXAMPLE_FILES.map((file) => fetch(file.value).then((res) => res.text())),
      );
      for (let i = 0; i < texts.length; i++) {
        if (cancelled) return;
        const { material } = await parseNtc(texts[i]);
        if (cancelled) return;
        setSlots((prev) => {
          const next = [...prev];
          next[i] = { label: EXAMPLE_FILES[i].label, material };
          return next;
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 p-6">
      <div className="flex flex-col gap-3 text-center">
        <h1 className="text-2xl font-semibold">Neural Texture Compression for Three.js</h1>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
          A three.js implementation of{' '}
          <a href={NTC_PAPER_URL} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-4">
            NVIDIA's 2023 Neural Texture Compression paper
          </a>
          . It represents materials as a tiny neural network instead of a stack of full-resolution textures,
          dramatically cutting memory and download size so scenes can carry much richer materials without
          blowing up their footprint.
        </p>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
          NTC compresses every texture of a material — albedo, normal, roughness, metalness, and more — together
          into a single representation, often smaller than just one texture of the original material. The
          materials below are each about 93KB.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Link to="/viewer">
          <Card className="h-full transition-colors hover:bg-accent">
            <CardHeader>
              <CardTitle className="text-primary underline underline-offset-4">Viewer</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Load and inspect .ntc material files interactively — rotate, zoom, and pan around a lit preview mesh.
              </p>
            </CardContent>
          </Card>
        </Link>

        <Link to="/trainer">
          <Card className="h-full transition-colors hover:bg-accent">
            <CardHeader>
              <CardTitle className="text-primary underline underline-offset-4">Trainer</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Fit a MaterialX material into a neural texture compression model, right in the browser, and export it
                as .ntc.
              </p>
            </CardContent>
          </Card>
        </Link>
      </div>

      <div className="relative aspect-square overflow-hidden rounded-xl border border-border bg-black">
        <NTCGridViewer slots={slots} />
      </div>
    </div>
  );
}
