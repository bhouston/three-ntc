import { SupportedTools } from './ToolApp';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from 'next-themes';
import { Toaster } from '@/components/ui/sonner';
import { ViewerPage } from '@/pages/ViewerPage';
import { TrainerPage } from '@/pages/TrainerPage';
import { GalleryPage } from '@/pages/GalleryPage';
import './styles.css';

const view = new URLSearchParams(location.search).get('view') ?? 'viewer';
const pages = { viewer: ViewerPage, trainer: TrainerPage, gallery: GalleryPage };

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
          <SupportedTools view={view} />
        </main>
      </div>
      <Toaster />
    </ThemeProvider>
  </StrictMode>,
);
