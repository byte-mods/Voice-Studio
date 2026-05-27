import { test, expect } from "@playwright/test";

/**
 * Happy-path smoke test.
 *
 * Walks: dashboard → create project → create dataset → submit noop job →
 * job detail page → see live log "Job finished".
 *
 * Requirements before running:
 *   - Backend running at NEXT_PUBLIC_API_BASE (default http://localhost:8000).
 *   - OAS_AUTH_REQUIRED unset (or true with a seeded user).
 */

const slug = (prefix: string) => `${prefix}-${Math.random().toString(36).slice(2, 8)}`;

test("smoke: create project, dataset, submit noop job, see log", async ({ page }) => {
  const projectSlug = slug("e2e");
  const datasetSlug = slug("ds");

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Welcome to Open Audio Studio/i })).toBeVisible();

  // Create project
  await page.goto("/projects");
  await page.getByLabel("Name").fill(`E2E ${projectSlug}`);
  await page.getByLabel("Slug").fill(projectSlug);
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.getByText(`E2E ${projectSlug}`)).toBeVisible();

  // Create dataset
  await page.goto("/datasets");
  await page.getByRole("button", { name: /New dataset/i }).click();
  await page.getByLabel("Name").fill(`Demo ${datasetSlug}`);
  await page.getByLabel("Slug").fill(datasetSlug);
  await page.getByRole("button", { name: /^Create$/ }).click();
  await expect(page.getByText(`Demo ${datasetSlug}`)).toBeVisible({ timeout: 5_000 });

  // Submit a noop job via the API directly (no UI form exists for arbitrary kinds yet).
  const projects = await page.request.get("/api/projects").then((r) => r.json());
  const project = projects.find((p: { slug: string }) => p.slug === projectSlug);
  const job = await page.request
    .post("/api/jobs", {
      data: { project_id: project.id, kind: "noop", name: "e2e-noop", config: {} },
    })
    .then((r) => r.json());

  // Job detail: wait for live log to print "Job finished"
  await page.goto(`/jobs/${job.id}`);
  await expect(page.getByText(/Job finished/)).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText("succeeded").first()).toBeVisible();
});
