import { createRoot } from 'react-dom/client';
import { SupportedTools } from './ToolApp';
import { ToolHostContext } from './lib/tool-host';
import { Toaster } from './components/ui/sonner';
import './styles.css';

export function mountTool(container: HTMLElement, view: string) {
  container.classList.add('dark');
  const root = createRoot(container);
  root.render(
    <ToolHostContext.Provider value={container}>
      <div className="dark flex min-h-[650px] flex-col bg-background text-foreground antialiased">
        <SupportedTools view={new URLSearchParams(location.search).get('view') ?? view} />
        <Toaster theme="dark" />
      </div>
    </ToolHostContext.Provider>,
  );
  return () => root.unmount();
}
