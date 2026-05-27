import { test, expect } from "@playwright/test";

/**
 * Verifies the login → app loop and the 401 → /login redirect.
 *
 * Skipped unless OAS_AUTH_REQUIRED=true on the backend; in anonymous mode
 * the studio doesn't need a session.
 */

const email = `e2e+${Date.now()}@local.test`;
const password = "playwright-password";

test.describe("auth required", () => {
  test.skip(
    !process.env.OAS_AUTH_REQUIRED_E2E,
    "set OAS_AUTH_REQUIRED_E2E=1 to run (server must also have OAS_AUTH_REQUIRED=true)",
  );

  test("signup then land on dashboard", async ({ page }) => {
    await page.goto("/login");
    await page.getByRole("button", { name: /Sign up/ }).click();
    await page.getByLabel("Email").fill(email);
    await page.getByLabel("Password").fill(password);
    await page.getByRole("button", { name: /Create account/ }).click();
    await expect(page.getByRole("heading", { name: /Welcome/ })).toBeVisible();
    await expect(page.getByText(email)).toBeVisible();
  });

  test("401 bounces to login with ?next preserved", async ({ page, context }) => {
    await context.clearCookies();
    await page.evaluate(() => window.localStorage.removeItem("oas_token"));
    await page.goto("/projects");
    await page.waitForURL(/\/login\?next=%2Fprojects/);
  });
});
