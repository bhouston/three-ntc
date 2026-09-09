import { createRootRoute, HeadContent, Link, Outlet, Scripts } from '@tanstack/react-router';
import { GithubIcon, HeartIcon } from 'lucide-react';
import { ThemeProvider } from 'next-themes';
import { GoogleAnalytics } from 'tanstack-router-ga4';

import { Toaster } from '@/components/ui/sonner';

const GITHUB_URL = 'https://github.com/bhouston/three-ntc';
const NPM_URL = 'https://www.npmjs.com/package/three-ntc';
const GA_MEASUREMENT_ID = 'G-6FBDBR436E';

function NpmIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
      <path d="M0 0v24h24V0H0zm19.2 19.2h-4.8V8.4H9.6v10.8H4.8V4.8h14.4v14.4z" />
    </svg>
  );
}

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
      <header className="flex items-center justify-between gap-4 border-b border-border px-6 py-3">
        <div className="flex items-center gap-4">
          <span className="font-semibold">three-ntc</span>
          <nav className="flex gap-4 text-sm">
            <Link to="/" className="text-primary underline underline-offset-4" activeProps={{ className: 'font-semibold' }}>
              Home
            </Link>
            <Link
              to="/viewer"
              className="text-primary underline underline-offset-4"
              activeProps={{ className: 'font-semibold' }}
            >
              Viewer
            </Link>
            <Link
              to="/trainer"
              className="text-primary underline underline-offset-4"
              activeProps={{ className: 'font-semibold' }}
            >
              Trainer
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-3 text-muted-foreground">
          <a href={GITHUB_URL} aria-label="GitHub" className="hover:text-foreground">
            <GithubIcon className="size-5" aria-hidden />
          </a>
          <a href={NPM_URL} aria-label="npm" className="hover:text-foreground">
            <NpmIcon className="size-5" />
          </a>
        </div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
      <footer className="border-t border-border">
        <p className="flex items-center justify-center gap-1 px-4 py-3 text-sm text-muted-foreground">
          Made by
          <a href="https://ben3d.ca" className="text-primary underline underline-offset-4">
            Ben Houston
          </a>
          with
          <HeartIcon className="size-3.5 fill-current text-destructive" aria-hidden />
          <span className="sr-only">love</span>
          — sponsored by
          <a href="https://landofassets.com" className="text-primary underline underline-offset-4">
            Land of Assets
          </a>
        </p>
      </footer>
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
        {import.meta.env.PROD ? <GoogleAnalytics measurementId={GA_MEASUREMENT_ID} /> : null}
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
          <Toaster />
        </ThemeProvider>
        <Scripts />
      </body>
    </html>
  );
}
