# Frontend — CNC Sentinel

Interactive operator dashboard for CNC cutting-tool wear visualization and Remaining Useful Life (RUL) advisory.

---

## Assigned Contributor: P1 (Lead / Advanced Motion & Integration) & P4 (Support / Assets)

### Stack:
- **Framework**: React 18+ with TypeScript
- **Bundler & Dev Server**: Vite
- **Styling**: Tailwind CSS v3 (`tailwindcss@^3.4.1`, `postcss`, `autoprefixer`)
- **Icons & Animation**: Lucide React, Framer Motion (or CSS keyframe motion primitives), Recharts

---

## 1. Initial Setup Commands

Run the following commands inside `frontend/` to initialize:

```bash
# In frontend/ directory:
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss@^3.4.1 postcss autoprefixer lucide-react recharts
npx tailwindcss init -p
```

### Configure Tailwind (`tailwind.config.js`):
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### Configure Directives in `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

## 2. Vite Development Proxy (Eliminating CORS & Hard-Coded Ports)

To avoid hard-coding `http://localhost:8000` and eliminate cross-origin request issues, configure the `/api` proxy in `vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
```

All frontend API calls should use relative paths (e.g. `fetch('/api/health')` or `fetch('/api/tools')`).

---

## 3. Frontend-Only Development Fixtures Policy

- While the backend model is in `model_status: "UNAVAILABLE"`, the frontend may use clearly labeled development fixtures located in `src/fixtures/mockFixtures.ts` to build out animations, gauge dials, and wear charts.
- **Strict Rule**: The production backend must **never** serve fabricated mock predictions. Mock fixtures are strictly restricted to frontend local sandbox testing.
- When toggled to "Live Mode", the frontend must strictly reflect the contract from [`../shared/api_contract.md`](../shared/api_contract.md).

---

## 4. UI Architecture & Views

- **System Status Bar**: Displays backend connection status, active tool case, and model verification indicator.
- **Tool Case & Cut Selector**: Scrub through cut passes ($1 \dots N$) dynamically populated from `/api/tools`.
- **Tool Wear Progression Chart**: Displays measured points (intermittent optical measurements) alongside model predictions and shaded uncertainty bands.
- **RUL & Decision Badge**: Prominently features the recommendation badge (`CONTINUE`, `INSPECT_SOON`, `REPLACE_TOOL`, or `MODEL_UNAVAILABLE`) with explanation tooltip.
