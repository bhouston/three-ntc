import { useEffect, useState } from 'react';
import { ViewerPage } from '@/pages/ViewerPage';
import { TrainerPage } from '@/pages/TrainerPage';
import { GalleryPage } from '@/pages/GalleryPage';
const pages = { viewer: ViewerPage, trainer: TrainerPage, gallery: GalleryPage };
type GPUAccess = { requestAdapter: () => Promise<unknown | null> };

export function SupportedTools({ view = 'viewer' }: { view?: string }) {
  const Page = pages[view as keyof typeof pages] ?? ViewerPage;
  const [supported, setSupported] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    const gpu = (navigator as Navigator & { gpu?: GPUAccess }).gpu;
    void (gpu?.requestAdapter() ?? Promise.resolve(null))
      .then((adapter) => {
        if (!cancelled) setSupported(Boolean(adapter));
      })
      .catch(() => {
        if (!cancelled) setSupported(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  if (supported === null) return <p className="p-6">Checking WebGPU availability…</p>;
  if (!supported)
    return (
      <p role="alert" className="p-6">
        These interactive tools require WebGPU. Use a WebGPU enabled browser on a supported device. Documentation
        remains available outside this tool.
      </p>
    );
  return <Page />;
}
