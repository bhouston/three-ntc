import { StrictMode, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from 'next-themes';
import { Toaster } from '@/components/ui/sonner';
import { ViewerPage } from '@/pages/ViewerPage';
import { TrainerPage } from '@/pages/TrainerPage';
import { GalleryPage } from '@/pages/GalleryPage';
import './styles.css';

const view = new URLSearchParams(location.search).get('view') ?? 'viewer';
const pages = { viewer: ViewerPage, trainer: TrainerPage, gallery: GalleryPage };
const Page = pages[view as keyof typeof pages] ?? ViewerPage;

type GPUAccess = { requestAdapter: () => Promise<unknown | null> };

function SupportedTools() {
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

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="flex min-h-screen flex-col bg-background text-foreground antialiased">
        <nav aria-label="Interactive tools" className="flex flex-wrap gap-4 border-b border-border px-4 py-3">
          {Object.keys(pages).map((name) => (
            <a
              key={name}
              href={`?view=${name}`}
              aria-current={name === view ? 'page' : undefined}
              className="text-sm capitalize underline underline-offset-4"
            >
              {name}
            </a>
          ))}
          <span className="ml-auto text-sm text-muted-foreground">WebGPU required for training</span>
        </nav>
        <main className="flex flex-1 flex-col">
          <SupportedTools />
        </main>
      </div>
      <Toaster />
    </ThemeProvider>
  </StrictMode>,
);
