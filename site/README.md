# Store Sales model comparison website

A static website (HTML + CSS + JavaScript) with no build step and no backend.
The whole `site/` folder is what gets deployed.

```
site/
  index.html          single page
  assets/style.css    styles (light and dark mode)
  assets/app.js       charts, cards, glossary, tooltips and the EN / TH switch (plain JS, no libraries)
  assets/i18n.js      Thai translations of the page text (English lives in index.html)
  assets/data.js      model results (RMSLE, error by forecast day, search results)
  Dockerfile          nginx image that serves the files
  nginx.conf          nginx settings (gzip, cache headers)
  docker-compose.yml  Compose file for Dokploy
```

The Inter and Noto Sans Thai fonts are loaded from Google Fonts; if they cannot be loaded the system font is used instead.

## Languages (EN / TH)

The page opens in **English**. The EN / ไทย switch in the top bar changes the whole page, and the choice is remembered in the browser.
A link can force a language with `?lang=th` or `?lang=en`.

- Page text: the English is in `index.html`; each translatable element has a `data-i18n="key"` attribute, and the Thai for that key is in `assets/i18n.js`.
- Text built in JavaScript (charts, model cards, glossary, table): both languages sit next to each other in `assets/app.js`, as `L("English", "ไทย")`.
- To add or change a sentence, edit the English in `index.html` and the matching key in `assets/i18n.js`. A key missing from `i18n.js` simply falls back to English.

## Run locally

Open `site/index.html` directly, or start a temporary server:

```
python -m http.server 3011 --directory site
```

Then open http://localhost:3011

## Deploy

| Where | How |
|---|---|
| **Netlify** | Drag the `site/` folder onto app.netlify.com/drop, or connect the repo and set Publish directory to `site` (leave the build command empty) |
| **Vercel** | `npx vercel site --prod`, or connect the repo and set Output Directory to `site` (Framework = Other) |
| **GitHub Pages** | Push to GitHub and deploy the `site/` folder with GitHub Actions (Settings → Pages → Source = GitHub Actions), or copy the contents of `site/` into `docs/` and choose Source = `docs` |
| **Cloudflare Pages** | Connect the repo and set Build output directory to `site` (no build command) |

## Deploy on Dokploy (Docker Compose)

Files used: `Dockerfile` (nginx:alpine serving the static files), `nginx.conf`, `docker-compose.yml` and `.dockerignore`.
The image is small, has no database, and publishes no host ports; Dokploy's Traefik reaches it.

1. Put this folder in a Git repository (GitHub, GitLab, Gitea, ...) and push
2. In Dokploy: **Create Service → Compose**
3. **Provider**: pick your Git provider, then the repository and branch
4. **Compose Path**: `./docker-compose.yml` if `site/` is the root of the repository,
   or `./site/docker-compose.yml` if the repository is the whole project
5. **Compose Type**: Docker Compose
6. **Domains** tab → Add Domain: Service Name = `web`, Port = `3011`, enter your domain and enable HTTPS (Let's Encrypt)
7. Click **Deploy**

Notes
- `docker-compose.yml` refers to the external network `dokploy-network`, which exists on a server where Dokploy is installed.
  On a machine without that network, use the commands below instead.
- Do not add `ports:` or `container_name:` to the compose file (Dokploy manages these).
- To update the numbers, edit `assets/data.js` and push again (turn on Auto Deploy in Dokploy to redeploy on every push).

Try the image locally (requires Docker):

```
docker build -t sales-site ./site
docker run --rm -p 3011:3011 sales-site
```

Then open http://localhost:3011

## Updating the numbers

All numbers are in `assets/data.js` (`window.RESULTS`). If you retrain the models, regenerate this file from the new results and deploy again.
