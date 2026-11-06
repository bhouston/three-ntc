import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Home() {
  return (
    <Layout
      title="Neural texture compression for three.js"
      description="Load compact neural materials, train your own in the browser, and learn how neural texture compression works."
    >
      <main>
        <section className="ntc-hero">
          <div className="container ntc-hero-grid">
            <div>
              <p className="ntc-eyebrow">NEURAL TEXTURE COMPRESSION / THREE.JS</p>
              <h1>
                Rich materials.
                <br />
                <span>Smaller downloads.</span>
              </h1>
              <p className="ntc-lead">
                One compact model for an entire PBR texture set. Decode it on the GPU and use it as a standard three.js
                node material.
              </p>
              <div className="ntc-actions">
                <Link className="button button--primary button--lg" to="/demo">
                  Explore the demo →
                </Link>
                <Link className="button button--outline button--secondary button--lg" to="/docs/getting-started">
                  Start building
                </Link>
              </div>
              <code className="ntc-install">npm install three three-ntc</code>
            </div>
            <figure className="ntc-materials">
              <img
                src={useBaseUrl('/img/materials.png')}
                alt="Eight neural materials rendered on spheres, showing metal, stone, brick and decorative surfaces"
              />
              <figcaption>Real materials from the interactive three-ntc viewer.</figcaption>
            </figure>
          </div>
        </section>
        <section className="container ntc-section">
          <p className="ntc-eyebrow">FROM TEXTURES TO NEURAL MATERIALS</p>
          <h2>A complete workflow, in your browser.</h2>
          <div className="ntc-cards">
            <article>
              <span>01 / EXPLORE</span>
              <h3>See the tradeoffs</h3>
              <p>Inspect sample materials, change sampling modes, and load your own .ntc files in the viewer.</p>
              <Link to="/demo">Open the demo →</Link>
            </article>
            <article>
              <span>02 / TRAIN</span>
              <h3>Make a neural material</h3>
              <p>Bake a MaterialX material, fit its latent grids and decoder, then export a portable .ntc asset.</p>
              <Link to="/trainer">Launch the trainer →</Link>
            </article>
            <article>
              <span>03 / UNDERSTAND</span>
              <h3>Look inside the model</h3>
              <p>Learn how learned feature grids, a small MLP and GPU decoding reconstruct a PBR texture set.</p>
              <Link to="/docs/theory">Read the explainer →</Link>
            </article>
          </div>
        </section>
        <section className="container ntc-section ntc-code-section">
          <div>
            <h2>
              A familiar material.
              <br />A different way to store it.
            </h2>
            <p>
              The runtime loads the model and connects the decoded channels to a node material. The trainer is a
              separate package, so it stays out of your production app.
            </p>
            <Link to="/docs/runtime">Runtime integration →</Link>
          </div>
          <pre>
            <code>{`import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

const { cpuModel, channelClassification } =
  await new NTCLoader().loadAsync('material.ntc');

mesh.material = new NTCNodeMaterial(
  cpuModel,
  channelClassification,
);`}</code>
          </pre>
        </section>
      </main>
    </Layout>
  );
}
