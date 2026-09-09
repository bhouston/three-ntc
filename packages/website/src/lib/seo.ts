const SITE_URL = 'https://three-ntc.ben3d.ca';
export const OG_IMAGE_URL = `${SITE_URL}/og-image.png`;

/** Builds title/description + OpenGraph/Twitter card meta tags for a route's `head()`. */
export function seoMeta({ title, description, path = '' }: { title: string; description: string; path?: string }) {
  const url = `${SITE_URL}${path}`;
  return [
    { title },
    { name: 'description', content: description },
    { property: 'og:type', content: 'website' },
    { property: 'og:site_name', content: 'three-ntc' },
    { property: 'og:title', content: title },
    { property: 'og:description', content: description },
    { property: 'og:image', content: OG_IMAGE_URL },
    { property: 'og:url', content: url },
    { name: 'twitter:card', content: 'summary_large_image' },
    { name: 'twitter:title', content: title },
    { name: 'twitter:description', content: description },
    { name: 'twitter:image', content: OG_IMAGE_URL },
  ];
}
