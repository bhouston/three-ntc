import { useEffect, useRef, useState } from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

type ToolModule = { mountTool: (container: HTMLElement, view: string) => () => void };

/** Mount the local React tools directly, isolating Tailwind from the docs theme. */
export default function ToolEmbed({ view }: { view: 'viewer' | 'trainer' | 'gallery' }) {
  const host = useRef<HTMLDivElement>(null);
  const tools = useBaseUrl('/tools/');
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    let unmount: (() => void) | undefined;
    const shadow = host.current!.shadowRoot ?? host.current!.attachShadow({ mode: 'open' });
    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = `${tools}embedded.css`;
    const container = document.createElement('div');
    shadow.replaceChildren(stylesheet, container);
    // The Vite entry owns its React root and GPU dependencies; it is loaded locally.
    void (import(/* webpackIgnore: true */ `${tools}embedded.js`) as Promise<ToolModule>)
      .then((module) => {
        if (!cancelled) unmount = module.mountTool(container, view);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
      unmount?.();
      shadow.replaceChildren();
    };
  }, [tools, view]);
  return (
    <div className="ntc-tool-embed">
      <div ref={host} />
      {error && <p role="alert">Unable to load the tool: {error}</p>}
    </div>
  );
}
