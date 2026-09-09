import { createRootRoute, HeadContent, Link, Outlet, Scripts } from '@tanstack/react-router';
import { ThemeProvider } from 'next-themes';

import { Toaster } from '@/components/ui/sonner';

import appCss from '@/styles.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'three-ntc demo' },
      { name: 'description', content: 'Neural Texture Compression viewer and MaterialX trainer demo for three-ntc.' },
    ],
    links: [{ rel: 'stylesheet', href: appCss }],
  }),
  shellComponent: RootDocument,
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="flex min-h-svh flex-col">
      <header className="flex items-center gap-4 border-b border-border px-6 py-3">
        <span className="font-semibold">three-ntc</span>
        <nav className="flex gap-4 text-sm text-muted-foreground">
          <Link to="/" activeProps={{ className: 'text-foreground' }}>
            Viewer
          </Link>
          <Link to="/trainer" activeProps={{ className: 'text-foreground' }}>
            Trainer
          </Link>
        </nav>
      </header>
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
    </div>
  );
}

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="bg-background text-foreground antialiased">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          {children}
          <Toaster />
        </ThemeProvider>
        <Scripts />
      </body>
    </html>
  );
}
