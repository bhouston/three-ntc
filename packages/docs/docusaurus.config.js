/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'three-ntc',
  tagline: 'Rich materials. Smaller downloads.',
  favicon: 'img/mark.svg',
  url: process.env.SITE_URL ?? 'https://bhouston.github.io',
  baseUrl: process.env.BASE_URL ?? '/three-ntc/',
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
    image: 'img/materials.png',
    navbar: {
      title: 'three-ntc',
      logo: { alt: '', src: 'img/mark.svg' },
      items: [
        { to: '/docs/getting-started', label: 'Docs', position: 'left' },
        { to: '/docs/api/runtime/', label: 'API', position: 'left' },
        { to: '/demo', label: 'Demo', position: 'left' },
        { to: '/trainer', label: 'Trainer', position: 'left' },
        { to: '/docs/theory', label: 'Theory', position: 'left' },
        { href: 'https://github.com/bhouston/three-ntc', label: 'GitHub', position: 'right' },
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
      copyright: `Copyright © ${new Date().getFullYear()} Ben Houston · MIT · Built with Docusaurus`,
    },
    prism: { additionalLanguages: ['bash', 'json'] },
  },
};
module.exports = config;
