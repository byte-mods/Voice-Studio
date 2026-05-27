import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config.
 *
 * The test suite assumes the FastAPI backend is reachable at
 * NEXT_PUBLIC_API_BASE (default http://localhost:8000) and that the studio is
 * in anonymous mode (or that you've pre-created a user — see e2e/login.spec.ts).
 *
 * The Next.js dev server is started automatically via `webServer` below; the
 * Python backend must be started separately (e.g. `make dev-server`).
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
