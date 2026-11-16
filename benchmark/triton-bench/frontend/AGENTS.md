# Prototype Instructions

The user selected the second displayed concept on 2026-11-13: warm light sidebar,
orange accent, vertical three-step submission flow. Use real same-origin API calls;
never fabricate scores, device health, users, quotas or successful NPU execution.
The Python frontend_gateway is the local browser entry. Build output is dist/client.

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

User feedback 2026-11-13: prioritize clear actions and plain Chinese. Reveal configuration after ZIP validation; lead results with verdict and speed; keep events, fingerprints and infrastructure in secondary disclosures. Preserve the warm sidebar visual system.

User feedback 2026-11-14: deliver a usable remote execution service, not a disabled preview. Provide a real multi-operator ZIP prominently; Abs is only an optional minimal format example. Explain scoring exclusions in plain language without treating unscored as failed.

User feedback 2026-11-14 (first-use UX): task lists and overview must summarize multi-operator runs by count and mode, never dump every operator name. Put full names in a deliberate disclosure on the detail page. Keep copy short, direct, and relevant to the next action or interpretation of a result. Hide operator/hosting limitations, frequency details, implementation guarantees and certification boilerplate in contextual details; retain real score coverage and actionable measurement caveats. Do not replace redundant copy with more redundant copy.
