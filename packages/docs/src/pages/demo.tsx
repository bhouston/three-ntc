import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Demo() {
  const tools = useBaseUrl('/tools/');
  return (
    <Layout
      title="Material demo"
      description="Explore neural materials and inspect your own .ntc files in the browser."
    >
      <main className="container ntc-tool-page">
        <h1>Explore neural materials</h1>
        <p>
          Inspect the gallery, open the single-material viewer, or drop in your own .ntc asset. Rendering requires a
          browser with hardware GPU support. <Link to="/docs/runtime">Read the runtime guide.</Link>
        </p>
        <iframe
          title="Interactive neural material gallery and viewer"
          src={`${tools}?view=gallery`}
          className="ntc-tool-frame"
          allow="fullscreen"
        />
        <p>
          <a href={`${tools}?view=viewer`} target="_blank" rel="noreferrer">
            Open the viewer in a full window ↗
          </a>
        </p>
      </main>
    </Layout>
  );
}
