import { defineCommand } from 'yargs-file-commands';

// STUB: `three-ntc train <input>` should run the three-ntc-trainer against a
// MaterialX file (or a directory of baked textures) headlessly and write out
// a .ntc file. This needs a WebGPU device outside the browser (e.g. via the
// `webgpu` npm package / Node's experimental WebGPU support) to run the
// trainer's compute shaders, which three-ntc-trainer doesn't set up yet.
// Left unimplemented until we settle on the cleanest way to do headless
// WebGPU training.
export const command = defineCommand({
  command: 'train <input>',
  describe: 'Train a .ntc model from a MaterialX file (not yet implemented)',
  builder: (yargs) =>
    yargs
      .positional('input', {
        describe: 'Path to a .mtlx MaterialX file',
        type: 'string',
        demandOption: true,
      })
      .option('output', {
        alias: 'o',
        describe: 'Path to write the .ntc file to',
        type: 'string',
      })
      .option('profile', {
        describe: 'Training profile (e.g. mobile-fast, mobile-balanced, desktop-quality)',
        type: 'string',
        default: 'mobile-balanced',
      }),
  handler: async (argv) => {
    throw new Error(
      `three-ntc train is not implemented yet. Would train "${argv.input}" ` +
        `with profile "${argv.profile}" and write to "${argv.output ?? '<input>.ntc'}". ` +
        `Headless WebGPU training is not wired up yet.`,
    );
  },
});
