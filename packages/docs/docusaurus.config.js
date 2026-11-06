const url = process.env.SITE_URL ?? 'https://three-ntc.ben3d.ca';
const baseUrl = process.env.BASE_URL ?? '/';
const githubIcon =
  '<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style="display:block"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>';
const link = (attributes) => ({ tagName: 'link', attributes });
// Raw head tags: `themeConfig.metadata` goes through Helmet, which keeps one tag per name and would drop a theme-color.
const themeColor = (content, scheme) => ({
  tagName: 'meta',
  attributes: { name: 'theme-color', content, media: `(prefers-color-scheme: ${scheme})` },
});

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'three-ntc',
  tagline: 'Rich materials. Smaller downloads.',
  favicon: 'img/mark.svg',
  url,
  baseUrl,
  headTags: [
    link({ rel: 'icon', type: 'image/png', sizes: '32x32', href: `${baseUrl}img/favicon-32x32.png` }),
    link({ rel: 'apple-touch-icon', sizes: '180x180', href: `${baseUrl}img/apple-touch-icon.png` }),
    link({ rel: 'manifest', href: `${baseUrl}site.webmanifest` }),
    themeColor('#ffffff', 'light'),
    themeColor('#101919', 'dark'),
    {
      tagName: 'script',
      attributes: { type: 'application/ld+json' },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'three-ntc',
        url: new URL(baseUrl, url).href,
        description:
          'Load compact neural materials, train your own in the browser, and learn how neural texture compression works.',
      }),
    },
  ],
  organizationName: 'bhouston',
  projectName: 'three-ntc',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: { format: 'md', hooks: { onBrokenMarkdownLinks: 'throw' } },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/bhouston/three-ntc/edit/main/packages/docs/',
        },
        blog: false,
        theme: { customCss: './src/css/custom.css' },
      },
    ],
  ],
  themeConfig: {
    image: 'img/social-card.png',
    metadata: [
      { property: 'og:site_name', content: 'three-ntc' },
      { name: 'twitter:site', content: '@BenHouston3D' },
    ],
    colorMode: { disableSwitch: true, respectPrefersColorScheme: true },
    navbar: {
      title: 'three-ntc',
      logo: { alt: '', src: 'img/mark.svg' },
      items: [
        { to: '/docs/getting-started', label: 'Docs', position: 'left' },
        { to: '/docs/api/runtime/', label: 'API', position: 'left' },
        { to: '/demo', label: 'Demo', position: 'left' },
        { to: '/trainer', label: 'Trainer', position: 'left' },
        { to: '/docs/theory', label: 'Theory', position: 'left' },
        {
          href: 'https://github.com/bhouston/three-ntc',
          html: githubIcon,
          'aria-label': 'GitHub repository',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            { label: 'Get started', to: '/docs/getting-started' },
            { label: 'How NTC works', to: '/docs/theory' },
          ],
        },
        {
          title: 'Build',
          items: [
            { label: 'Runtime API', to: '/docs/api/runtime/' },
            { label: 'Trainer API', to: '/docs/api/trainer/' },
          ],
        },
        {
          title: 'Project',
          items: [
            { label: 'GitHub', href: 'https://github.com/bhouston/three-ntc' },
            { label: 'npm', href: 'https://www.npmjs.com/package/three-ntc' },
          ],
        },
      ],
      copyright:
        'Made with ❤️ by <a href="https://ben3d.ca">Ben Houston</a>. Sponsored by <a href="https://landofassets.com">Land of Assets</a>',
    },
    prism: { additionalLanguages: ['bash', 'json'] },
  },
};
module.exports = config;
