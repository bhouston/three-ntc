import { useCallback, useEffect, useState } from 'react';

// Keep shareable material URLs without requiring a server-side router.
export function useSourceSearch() {
  const [src, setSrc] = useState<string | null | undefined>(() => new URLSearchParams(location.search).get('src') ?? undefined);
  useEffect(() => {
    const update = () => setSrc(new URLSearchParams(location.search).get('src') ?? undefined);
    addEventListener('popstate', update);
    return () => removeEventListener('popstate', update);
  }, []);
  const navigate = useCallback(({ search }: { search: { src?: string } }) => {
    const url = new URL(location.href);
    if (search.src) url.searchParams.set('src', search.src);
    else url.searchParams.delete('src');
    history.pushState(null, '', url);
    // An uploaded file clears the URL without reloading the default example.
    setSrc(search.src ?? null);
  }, []);
  return { src, navigate };
}
