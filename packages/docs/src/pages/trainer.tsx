import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Trainer() {
  const tools = useBaseUrl('/tools/');
  return (
    <Layout title="Browser trainer" description="Train and export neural materials from MaterialX using WebGPU.">
      <main className="container ntc-tool-page">
        <h1>Train a neural material</h1>
        <p>
          Choose a MaterialX source, compare the reconstruction, and export the learned model. Training runs locally on
          your GPU and requires WebGPU over HTTPS or localhost.{' '}
          <Link to="/docs/training">Follow the training guide.</Link>
        </p>
        <iframe
          title="Neural texture compression trainer"
          src={`${tools}?view=trainer`}
          className="ntc-tool-frame"
          allow="fullscreen"
        />
        <p>
          <a href={`${tools}?view=trainer`} target="_blank" rel="noreferrer">
            Open the trainer in a full window ↗
          </a>
        </p>
      </main>
    </Layout>
  );
}
