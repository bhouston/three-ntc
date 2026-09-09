#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { fileCommands } from 'yargs-file-commands';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const main = async () => {
  const commandsDir = path.join(__dirname, 'commands');

  return yargs(hideBin(process.argv))
    .scriptName('three-ntc')
    .command(await fileCommands({ commandDirs: [commandsDir] }))
    .demandCommand(1)
    .strict()
    .help().argv;
};

main();
