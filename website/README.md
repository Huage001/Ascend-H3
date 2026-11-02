# Ascend-H3 project page

This is a standalone React/Vite presentation for the repository. It does not
change the inference stack or require an Ascend environment.

```bash
cd website
npm ci
npm run dev
```

Run `npm run build` to create the static site in `dist/`. The benchmark values
and their qualifications are drawn from the root README. The ten comparison
videos are stored in `public/videos` and their SHA256 values match the
[124-frame provenance](../examples/videos/qualitative-1344-124.json).
Run `npm run verify:videos` to check all ten files against `videos-manifest.json`.

The GitHub Pages workflow builds this site with the `/Ascend-H3/` base path.
In the repository's **Settings > Pages**, select **GitHub Actions** as the
publishing source. Changes under `website/` on `main` then trigger deployment.

The site uses i18next and react-i18next for Simplified Chinese and English.
Translations live in `src/locales/`; the language choice is saved in browser
local storage when available and defaults to Simplified Chinese.

The components in `src/react-bits` are sourced from the React Bits JS/CSS
registry, with small integration changes. Their separate MIT + Commons Clause
terms and copyright notice are in `src/react-bits/LICENSE.md`.
