import { describe, expect, it } from "vitest";

describe("Mappls configuration", () => {
  it("accepts the configured browser SDK token", async () => {
    const token = process.env.VITE_MAPPLS_TOKEN;
    expect(token, "VITE_MAPPLS_TOKEN must be configured").toBeTruthy();
    const response = await fetch(`https://apis.mappls.com/advancedmaps/api/${token}/map_sdk?v=3.0&layer=vector`, { method: "GET" });
    expect(response.ok, `Mappls SDK bootstrap returned ${response.status}`).toBe(true);
    expect(response.headers.get("content-type") ?? "").toMatch(/javascript|text/i);
  }, 20_000);
});
