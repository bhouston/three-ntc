import { createFileRoute } from '@tanstack/react-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { useGoogleAnalytics } from 'tanstack-router-ga4';

import { NTCViewer } from '@/components/NTCViewer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EXAMPLE_FILES, type LoadedMaterial, parseNtc } from '@/lib/ntc-examples';

export interface ViewerSearch {
  src?: string;
}

export const Route = createFileRoute('/viewer')({
  component: ViewerPage,
  validateSearch: (search: Record<string, unknown>): ViewerSearch => ({
    src: typeof search.src === 'string' ? search.src : undefined,
  }),
  head: () => ({
    meta: [{ title: 'Three-NTC Viewer' }],
  }),
});

function ViewerPage() {
  const ga = useGoogleAnalytics();
  const navigate = Route.useNavigate();
  const { src } = Route.useSearch();
  const [loaded, setLoaded] = useState<LoadedMaterial | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async (text: string, sourceLabel: string) => {
    setLoading(true);
    try {
      const result = await parseNtc(text);
      setLoaded(result);
    } catch (err) {
      console.error(err);
      toast.error(`Failed to load ${sourceLabel}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadFromUrl = useCallback(
    async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) {
        toast.error(`Failed to fetch ${url}`);
        return;
      }
      await load(await res.text(), url);
    },
    [load],
  );

  const loadFromFile = useCallback(
    async (file: File) => {
      await load(await file.text(), file.name);
      void navigate({ search: {} });
    },
    [load, navigate],
  );

  // ?src= drives the loaded material - a relative URL matches an example by
  // its `value`, a full URL loads external material shared via that link.
  // Falls back to the first example when the param is absent.
  useEffect(() => {
    void loadFromUrl(src ?? EXAMPLE_FILES[0].value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  return (
    <div className="grid flex-1 grid-cols-1 gap-4 p-4 md:grid-cols-[1fr_320px]">
      <div className="relative min-h-[320px] overflow-hidden rounded-xl border border-border bg-black">
        <NTCViewer material={loaded?.material ?? null} />
        {!loaded && !loading && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">
            Load a .ntc file to preview it
          </div>
        )}
      </div>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Load material</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <select
              value={src ?? EXAMPLE_FILES[0].value}
              onChange={(e) => {
                if (!e.target.value) return;
                ga.event('viewer-default-ntc', { ntc_file: e.target.value });
                void navigate({ search: { src: e.target.value } });
              }}
              className="h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {EXAMPLE_FILES.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>

            <button
              type="button"
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files[0];
                if (!file) return;
                ga.event('viewer-drop-ntc', { file_name: file.name });
                void loadFromFile(file);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ${
                dragOver ? 'border-primary bg-accent text-foreground' : 'border-border text-muted-foreground'
              }`}
            >
              Drag & drop a .ntc file here, or click to browse
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".ntc"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void loadFromFile(file);
                e.target.value = '';
              }}
            />

            <Button variant="outline" disabled>
              {loading ? 'Loading…' : 'Ready'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Material info</CardTitle>
          </CardHeader>
          <CardContent>
            {loaded ? (
              <dl className="flex flex-col gap-2 text-sm">
                <div>
                  <dt className="text-muted-foreground">Name</dt>
                  <dd>{loaded.name}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Active channels</dt>
                  <dd>{loaded.channels.length > 0 ? loaded.channels.join(', ') : '—'}</dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground">No material loaded yet.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
