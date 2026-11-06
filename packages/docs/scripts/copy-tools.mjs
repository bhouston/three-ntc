import { cp, access } from 'node:fs/promises';
await access('../playground/dist/index.html');
await cp('../playground/dist', process.argv.includes('--dev') ? 'static/tools' : 'build/tools', { recursive: true });
