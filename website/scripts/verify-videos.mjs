import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const manifest = JSON.parse(await readFile(new URL('../videos-manifest.json', import.meta.url), 'utf8'));
const directory = resolve(import.meta.dirname, '../public/videos');
const expectedFiles = ['full', 'vsa', 'veda', 'vdn', 'spark'].flatMap(method =>
  ['bf16', 'int8'].map(precision => `${method}-${precision}.mp4`)
);
if (manifest.length !== expectedFiles.length || expectedFiles.some(file => !manifest.some(video => video.file === file))) {
  throw new Error('Video manifest must contain each of the ten comparisons exactly once');
}

for (const video of manifest) {
  const path = resolve(directory, video.file);
  if (!path.startsWith(`${directory}/`)) throw new Error(`Invalid video path: ${video.file}`);
  const { size } = await stat(path);
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  if (size !== video.bytes || hash.digest('hex') !== video.sha256) {
    throw new Error(`Video verification failed: ${video.file}`);
  }
  console.log(`Verified ${video.file}`);
}
