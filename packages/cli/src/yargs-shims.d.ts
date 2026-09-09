// yargs@18's package.json only advertises `types` for its `./browser` export,
// not the main entrypoint or `./helpers` actually used here - ambient shim
// until that's fixed upstream.
declare module 'yargs';
declare module 'yargs/helpers';
