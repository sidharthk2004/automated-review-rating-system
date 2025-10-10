# Real‑Time Automated Review Rating System

**A web application that lets users submit reviews and instantly receive AI‑generated star ratings using sentiment analysis. Reviews are stored and displayed in real time.**

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Technology Stack](#technology-stack)
* [Screenshots](#screenshots)
* [Getting Started (Frontend)](#getting-started-frontend)
* [Suggested Backend & Real‑Time Architecture](#suggested-backend--real-time-architecture)
* [API Endpoints (Suggested)](#api-endpoints-suggested)
* [Environment Variables](#environment-variables)
* [Development](#development)
* [Build & Production](#build--production)
* [Testing & Linting](#testing--linting)
* [Deployment Suggestions](#deployment-suggestions)
* [Project Structure](#project-structure)


---

## Project Overview

This project is a front‑end React + TypeScript application (Vite) that allows users to write textual reviews and receive an automatic star rating computed by an AI model (sentiment analysis). Ratings and reviews propagate to all connected users in real time, enabling a responsive and interactive feedback experience.

Use cases include e‑commerce integrations, service review portals, internal feedback dashboards, and customer experience platforms.

---

## Key Features

* Write and submit reviews.
* Automated star ratings generated from review text using sentiment analysis.
* Real‑time updates: new reviews and rating breakdowns appear without page reloads.
* Review listing with sorting (newest / oldest / highest / lowest).
* Rating breakdown and aggregate score display.

---

## Technology Stack

**Frontend**

* React (Vite + TypeScript)
* TailwindCSS
* shadcn/ui components
* lucide‑react (icons)
* axios (HTTP client)
* react‑router (routing)

**Suggested Backend**

* Node.js + Express (or Fastify) with TypeScript
* WebSocket (Socket.IO) for real‑time events
* Database: MongoDB
* Optional: Redis for pub/sub when scaling
* AI: a sentiment model endpoint (self‑hosted or third‑party inference API)

---

## Screenshots

> Add the screenshots from `/src/assets` or a `screenshots/` folder and reference them here (replace placeholders below):

* `screenshots/reviewPage.png`
  <img width="903" height="450" alt="image" src="https://github.com/user-attachments/assets/8ad20c82-2de5-4134-b5ff-31f2d65a3f80" />

* `screenshots/writeReviewPage.png`
  <img width="940" height="562" alt="image" src="https://github.com/user-attachments/assets/a1b77be6-a3f6-4331-af05-b555bcaddea3" />

* `screenshots/ratingPage.png`
<img width="940" height="532" alt="image" src="https://github.com/user-attachments/assets/a5c41eed-616f-4b20-93cd-728a79aa3d19" />

* `screenshots/loading.png`
  <img width="940" height="547" alt="image" src="https://github.com/user-attachments/assets/f4ab5436-e427-44e1-af4e-e67d06d45b51" />


---

## Getting Started (Frontend)

### Prerequisites

* Node.js >= 18
* npm or pnpm

### Install & Run

```bash
# scaffolded by you
npm create vite@latest review-rating-system -- --template react-ts
cd review-rating-system

# install dependencies
npm install

# start dev server
npm run dev
```

The project assumes a backend API (see [API Endpoints](#api-endpoints-suggested)) and a real‑time socket server for push updates.

---



## API Endpoints (Suggested)

These are example endpoints the frontend expects to interact with. Update paths if your backend differs.

```
GET  /api/reviews?sort=newest&limit=20&page=1    # list reviews + metadata
POST /api/reviews                                # create a new review (body: { author?, text })
GET  /api/reviews/:id                             # get single review

```


## Environment Variables

Example `.env` / `.env.local` for the frontend (Vite):

```
VITE_API_BASE_URL=http://localhost3000/api
VITE_SOCKET_URL=http://localhost:5000
```



## Development Notes

* The React codebase uses strict TypeScript settings. Some compilation rules may require you to add small type guards or `unknown` assertions when integrating 3rd party libs.
* `useShowReviews` is a custom hook that handles fetching, sorting, pagination, and `displayedReviews` logic. Keep network and socket updating logic centralized in a hook or a context provider.
* Keep UI components small and reusable (e.g., `ReviewCard`, `TotalRatingCard`, `SortReviews`, `ReviewForm`).

---

## Build & Production

```bash
# build
npm run build

# preview (local static server)
npm run preview
```

For production, serve the built `dist/` from a static host or integrate into a full‑stack server. If the backend and frontend are on different origins, configure CORS and run a reverse proxy (Nginx) for nicer URLs.

---

## Testing & Linting

* Add unit and integration tests with your preferred tools (Vitest / Jest + React Testing Library).
* Lint with ESLint and TypeScript rules already included in `devDependencies`.

Example commands you can add to `package.json`:

```json
"scripts": {
  "test": "vitest",
  "lint": "eslint 'src/**/*.{ts,tsx}'",
  "format": "prettier --write ."
}
```

---

## Deployment Suggestions

* Frontend: Vercel, Netlify, or static hosting behind CDN.
* Backend: Heroku, Render, Railway, or containerized on AWS ECS / GCP Cloud Run.
* Use HTTPS, rate limiting, and authentication for APIs in production.
* If you rely on an external sentiment API, ensure credentials are kept secret and request quotas are handled gracefully.

---

## Project Structure (frontend)

```
src/
├─ assets/
├─ components/
│  ├─ ui/ (Button, Input, Loading, etc.)
├─ features/
│  └─ review/ (hooks, ReviewCard, ReviewForm, TotalRatingCard)
├─ layouts/
│  └─ MainLayout.tsx
├─ pages/
│  ├─ ReviewPage.tsx
│  └─ WriteReviewPage.tsx
├─ routes/
│  └─ AppRoutes.tsx
├─ main.tsx
└─ index.css
```

---



