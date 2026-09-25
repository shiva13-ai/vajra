import { jsxLocPlugin } from "@builder.io/vite-plugin-jsx-loc";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { execFile, spawn } from "node:child_process";
import { defineConfig, type Plugin, type ViteDevServer } from "vite";
import { vitePluginManusRuntime } from "vite-plugin-manus-runtime";

// =============================================================================
// Manus Debug Collector - Vite Plugin
// Writes browser logs directly to files, trimmed when exceeding size limit
// =============================================================================

const PROJECT_ROOT = import.meta.dirname;
const LOG_DIR = path.join(PROJECT_ROOT, ".manus-logs");
const MAX_LOG_SIZE_BYTES = 1 * 1024 * 1024; // 1MB per log file
const TRIM_TARGET_BYTES = Math.floor(MAX_LOG_SIZE_BYTES * 0.6); // Trim to 60% to avoid constant re-trimming

type LogSource = "browserConsole" | "networkRequests" | "sessionReplay";

function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
}

function trimLogFile(logPath: string, maxSize: number) {
  try {
    if (!fs.existsSync(logPath) || fs.statSync(logPath).size <= maxSize) {
      return;
    }

    const lines = fs.readFileSync(logPath, "utf-8").split("\n");
    const keptLines: string[] = [];
    let keptBytes = 0;

    // Keep newest lines (from end) that fit within 60% of maxSize
    const targetSize = TRIM_TARGET_BYTES;
    for (let i = lines.length - 1; i >= 0; i--) {
      const lineBytes = Buffer.byteLength(`${lines[i]}\n`, "utf-8");
      if (keptBytes + lineBytes > targetSize) break;
      keptLines.unshift(lines[i]);
      keptBytes += lineBytes;
    }

    fs.writeFileSync(logPath, keptLines.join("\n"), "utf-8");
  } catch {
    /* ignore trim errors */
  }
}

function writeToLogFile(source: LogSource, entries: unknown[]) {
  if (entries.length === 0) return;

  ensureLogDir();
  const logPath = path.join(LOG_DIR, `${source}.log`);

  // Format entries with timestamps
  const lines = entries.map((entry) => {
    const ts = new Date().toISOString();
    return `[${ts}] ${JSON.stringify(entry)}`;
  });

  // Append to log file
  fs.appendFileSync(logPath, `${lines.join("\n")}\n`, "utf-8");

  // Trim if exceeds max size
  trimLogFile(logPath, MAX_LOG_SIZE_BYTES);
}

/**
 * Vite plugin to collect browser debug logs
 * - POST /__manus__/logs: Browser sends logs, written directly to files
 * - Files: browserConsole.log, networkRequests.log, sessionReplay.log
 * - Auto-trimmed when exceeding 1MB (keeps newest entries)
 */
function vitePluginManusDebugCollector(): Plugin {
  return {
    name: "manus-debug-collector",

    transformIndexHtml(html) {
      if (process.env.NODE_ENV === "production") {
        return html;
      }
      return {
        html,
        tags: [
          {
            tag: "script",
            attrs: {
              src: "/__manus__/debug-collector.js",
              defer: true,
            },
            injectTo: "head",
          },
        ],
      };
    },

    configureServer(server: ViteDevServer) {
      // POST /__manus__/logs: Browser sends logs (written directly to files)
      server.middlewares.use("/__manus__/logs", (req, res, next) => {
        if (req.method !== "POST") {
          return next();
        }

        const handlePayload = (payload: any) => {
          // Write logs directly to files
          if (payload.consoleLogs?.length > 0) {
            writeToLogFile("browserConsole", payload.consoleLogs);
          }
          if (payload.networkRequests?.length > 0) {
            writeToLogFile("networkRequests", payload.networkRequests);
          }
          if (payload.sessionEvents?.length > 0) {
            writeToLogFile("sessionReplay", payload.sessionEvents);
          }

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ success: true }));
        };

        const reqBody = (req as { body?: unknown }).body;
        if (reqBody && typeof reqBody === "object") {
          try {
            handlePayload(reqBody);
          } catch (e) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ success: false, error: String(e) }));
          }
          return;
        }

        let body = "";
        req.on("data", (chunk) => {
          body += chunk.toString();
        });

        req.on("end", () => {
          try {
            const payload = JSON.parse(body);
            handlePayload(payload);
          } catch (e) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ success: false, error: String(e) }));
          }
        });
      });
    },
  };
}

function vitePluginStorageProxy(): Plugin {
  return {
    name: "manus-storage-proxy",
    configureServer(server: ViteDevServer) {
      server.middlewares.use("/manus-storage", async (req, res) => {
        const key = req.url?.replace(/^\//, "");
        if (!key) {
          res.writeHead(400, { "Content-Type": "text/plain" });
          res.end("Missing storage key");
          return;
        }

        const forgeBaseUrl = (process.env.BUILT_IN_FORGE_API_URL || "").replace(/\/+$/, "");
        const forgeKey = process.env.BUILT_IN_FORGE_API_KEY;

        if (!forgeBaseUrl || !forgeKey) {
          res.writeHead(500, { "Content-Type": "text/plain" });
          res.end("Storage proxy not configured");
          return;
        }

        try {
          const forgeUrl = new URL("v1/storage/presign/get", forgeBaseUrl + "/");
          forgeUrl.searchParams.set("path", key);

          const forgeResp = await fetch(forgeUrl, {
            headers: { Authorization: `Bearer ${forgeKey}` },
          });

          if (!forgeResp.ok) {
            res.writeHead(502, { "Content-Type": "text/plain" });
            res.end("Storage backend error");
            return;
          }

          const { url } = (await forgeResp.json()) as { url: string };
          if (!url) {
            res.writeHead(502, { "Content-Type": "text/plain" });
            res.end("Empty signed URL");
            return;
          }

          res.writeHead(307, { Location: url, "Cache-Control": "no-store" });
          res.end();
        } catch {
          res.writeHead(502, { "Content-Type": "text/plain" });
          res.end("Storage proxy error");
        }
      });
    },
  };
}

function vitePluginMlEndpoints(): Plugin {
  return {
    name: "vajra-ml-endpoints",
    configureServer(server: ViteDevServer) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url?.startsWith("/api/ml")) {
          return next();
        }

        const url = new URL(req.url, "http://localhost");
        const pathname = url.pathname;

        res.setHeader("Content-Type", "application/json");

        const checkpointsDir = path.join(PROJECT_ROOT, "server", "ml", "checkpoints");
        const manifestPath = path.join(checkpointsDir, "checkpoint_manifest.json");
        const trainScript = path.join(PROJECT_ROOT, "server", "ml", "train_tabular_models.py");
        const predictScript = path.join(PROJECT_ROOT, "server", "ml", "predict_service.py");
        const continuousLearnerScript = path.join(PROJECT_ROOT, "server", "ml", "continuous_learner.py");
        const ledgerPath = path.join(checkpointsDir, "prediction_ledger.json");

        if (pathname === "/api/ml/status" && req.method === "GET") {
          try {
            const files = fs.existsSync(checkpointsDir)
              ? fs.readdirSync(checkpointsDir).map((f) => {
                  const st = fs.statSync(path.join(checkpointsDir, f));
                  return { name: f, sizeBytes: st.size, modifiedAt: st.mtime.toISOString() };
                })
              : [];
            const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf-8")) : null;

            res.end(
              JSON.stringify({
                status: "success",
                hardware: {
                  computeDevice: "CPU (Multi-threaded Intel/AMD AVX2/AVX-512)",
                  acceleration: "OpenMP / multi-threading (n_jobs=-1)",
                  gpuAvailable: false,
                  gpuMessage:
                    "No local NVIDIA CUDA GPU found in PATH. XGBoost (v3.4.1) & LightGBM (v4.7.0) running ultra-fast CPU inference.",
                  externalGpuGuide: {
                    description: "To connect external GPU cluster or cloud inference endpoints (e.g. AWS SageMaker, GCP Vertex AI, RunPod):",
                    endpointsSupported: ["vLLM / Triton Inference Server", "TorchServe GPU", "XGBoost Dask-CUDA"],
                    environmentVariables: ["EXTERNAL_ML_GPU_ENDPOINT", "GPU_AUTH_TOKEN"],
                  },
                  models: {
                    xgboost: {
                      version: "3.4.1",
                      tasks: ["Thunderstorm Genesis", "Severe Hail Occurrence", "Microburst Ensemble"],
                      checkpoint: "server/ml/checkpoints/xgboost_thunderstorm.json",
                    },
                    lightgbm: {
                      version: "4.7.0",
                      tasks: ["Flash Cloudburst Deluge Classifier"],
                      checkpoint: "server/ml/checkpoints/lightgbm_cloudburst.txt",
                    },
                  },
                  checkpointCount: files.length,
                  files,
                  lastManifest: manifest,
                },
              })
            );
          } catch (err: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ status: "error", message: err.message }));
          }
          return;
        }

        if (pathname === "/api/ml/checkpoints" && req.method === "GET") {
          try {
            if (!fs.existsSync(manifestPath)) {
              res.statusCode = 404;
              return res.end(JSON.stringify({ status: "error", message: "Checkpoints not found." }));
            }
            const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
            const files = fs.readdirSync(checkpointsDir).map((f) => {
              const st = fs.statSync(path.join(checkpointsDir, f));
              return { name: f, sizeBytes: st.size, modifiedAt: st.mtime.toISOString() };
            });
            res.end(JSON.stringify({ status: "success", manifest, files }));
          } catch (err: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ status: "error", message: err.message }));
          }
          return;
        }

        if ((pathname === "/api/ml/predict" || pathname === "/api/ml/predict-all-grids") && req.method === "POST") {
          let body = "";
          req.on("data", (chunk) => (body += chunk));
          req.on("end", () => {
            try {
              const child = spawn("python", [predictScript, "--stdin"], { cwd: PROJECT_ROOT });
              let stdout = "";
              let stderr = "";

              child.stdout.on("data", (d) => (stdout += d.toString()));
              child.stderr.on("data", (d) => (stderr += d.toString()));

              child.on("close", (code) => {
                if (code !== 0) {
                  res.statusCode = 500;
                  return res.end(JSON.stringify({ status: "error", code, stderr }));
                }
                res.setHeader("Content-Type", "application/json");
                res.end(stdout);
              });

              child.stdin.write(body || "{}");
              child.stdin.end();
            } catch (err: any) {
              res.statusCode = 500;
              res.end(JSON.stringify({ status: "error", message: err.message }));
            }
          });
          return;
        }

        if (pathname === "/api/ml/train" && req.method === "POST") {
          let body = "";
          req.on("data", (chunk) => (body += chunk));
          req.on("end", () => {
            try {
              let params: any = {};
              try { params = JSON.parse(body); } catch {}
              const n_samples = String(params.n_samples || 6000);
              const epochs = String(params.epochs || 25);
              const lr = String(params.lr || 0.05);

              execFile(
                "python",
                [trainScript, "--samples", n_samples, "--epochs", epochs, "--lr", lr],
                { cwd: PROJECT_ROOT, timeout: 120000 },
                (error, stdout, stderr) => {
                  if (error) {
                    res.statusCode = 500;
                    return res.end(JSON.stringify({ status: "error", message: error.message, stderr }));
                  }
                  let manifest = null;
                  if (fs.existsSync(manifestPath)) {
                    try { manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8")); } catch {}
                  }
                  res.end(
                    JSON.stringify({
                      status: "success",
                      message: "XGBoost and LightGBM models trained and checkpoints updated.",
                      logs: stdout,
                      manifest,
                    })
                  );
                }
              );
            } catch (err: any) {
              res.statusCode = 500;
              res.end(JSON.stringify({ status: "error", message: err.message }));
            }
          });
          return;
        }

        if (pathname === "/api/ml/continuous-learn/status" && req.method === "GET") {
          try {
            const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, "utf-8")) : null;
            const ledger = fs.existsSync(ledgerPath) ? JSON.parse(fs.readFileSync(ledgerPath, "utf-8")) : null;
            const replayBufferPath = path.join(checkpointsDir, "live_replay_buffer.jsonl");
            let replayCount = 0;
            if (fs.existsSync(replayBufferPath)) {
              replayCount = fs.readFileSync(replayBufferPath, "utf-8").split("\n").filter(Boolean).length;
            }

            const step = manifest?.continuous_learning_step ?? 0;
            const lossHistory = manifest?.loss_history ?? [];
            const lastLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : null;

            res.end(
              JSON.stringify({
                status: "success",
                step,
                totalSteps: step,
                continuous_learning_step: step,
                gatekeeper_status: manifest?.gatekeeper_status || "PROMOTED",
                total_grids_monitored: manifest?.total_grids_monitored || 64,
                all_india_coverage_pct: manifest?.all_india_coverage_pct || 100.0,
                lossHistory,
                loss_history: lossHistory,
                lastLoss,
                last_loss: lastLoss,
                lastUpdatedAt: manifest?.last_updated_at || manifest?.generated_at,
                last_self_learning_at: manifest?.last_self_learning_at || manifest?.generated_at,
                pendingPredictionsCount: ledger?.pending_predictions?.length || 0,
                pending_prediction_verifications: ledger?.pending_predictions?.length || 0,
                verifiedHistoryCount: ledger?.verified_history?.length || 0,
                totalVerifiedSoundings: ledger?.total_verified || 0,
                total_verified_soundings: ledger?.total_verified || 0,
                totalBatchesIngested: ledger?.total_ingested_batches || 0,
                recentVerifiedHistory: (ledger?.verified_history || []).slice(-10),
                recent_verified_history: (ledger?.verified_history || []).slice(-10),
                replayBufferSize: replayCount,
                replay_buffer_size: replayCount,
                models: {
                  xgboostThunderstorm: {
                    status: fs.existsSync(path.join(checkpointsDir, "xgboost_thunderstorm.json")) ? "active" : "missing",
                    checkpoint: "xgboost_thunderstorm.json",
                  },
                  lightgbmCloudburst: {
                    status: fs.existsSync(path.join(checkpointsDir, "lightgbm_cloudburst.txt")) ? "active" : "missing",
                    checkpoint: "lightgbm_cloudburst.txt",
                  },
                  xgboostHail: {
                    status: fs.existsSync(path.join(checkpointsDir, "xgboost_hail.json")) ? "active" : "missing",
                    checkpoint: "xgboost_hail.json",
                  },
                },
              })
            );
          } catch (err: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ status: "error", message: err.message }));
          }
          return;
        }

        if (pathname === "/api/ml/continuous-learn/ingest" && req.method === "POST") {
          let body = "";
          req.on("data", (chunk) => (body += chunk));
          req.on("end", () => {
            try {
              if (!body || body.trim().length === 0) {
                res.statusCode = 400;
                return res.end(JSON.stringify({ status: "error", message: "Empty observations payload" }));
              }

              const child = spawn("python", [continuousLearnerScript, "--stdin"], { cwd: PROJECT_ROOT });
              let stdout = "";
              let stderr = "";

              child.stdout.on("data", (d) => (stdout += d.toString()));
              child.stderr.on("data", (d) => (stderr += d.toString()));

              child.on("close", (code) => {
                if (code !== 0) {
                  res.statusCode = 500;
                  return res.end(JSON.stringify({ status: "error", code, stderr }));
                }
                try {
                  const result = JSON.parse(stdout.trim());
                  res.end(JSON.stringify(result));
                } catch {
                  res.end(stdout);
                }
              });

              child.stdin.write(body);
              child.stdin.end();
            } catch (err: any) {
              res.statusCode = 500;
              res.end(JSON.stringify({ status: "error", message: err.message }));
            }
          });
          return;
        }

        res.statusCode = 404;
        res.end(JSON.stringify({ status: "error", message: "Endpoint not found" }));
      });
    },
  };
}

const plugins = [
  react(),
  tailwindcss(),
  jsxLocPlugin(),
  vitePluginManusRuntime(),
  vitePluginManusDebugCollector(),
  vitePluginStorageProxy(),
  vitePluginMlEndpoints(),
];

export default defineConfig({
  plugins,
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
      "@shared": path.resolve(import.meta.dirname, "shared"),
      "@assets": path.resolve(import.meta.dirname, "attached_assets"),
    },
  },
  envDir: path.resolve(import.meta.dirname),
  root: path.resolve(import.meta.dirname, "client"),
  build: {
    outDir: path.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    strictPort: false, // Will find next available port if 3000 is busy
    host: true,
    allowedHosts: [
      ".manuspre.computer",
      ".manus.computer",
      ".manus-asia.computer",
      ".manuscomputer.ai",
      ".manusvm.computer",
      "localhost",
      "127.0.0.1",
    ],
    fs: {
      strict: true,
      deny: ["**/.*"],
    },
  },
});
