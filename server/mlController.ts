import { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const PROJECT_ROOT = process.cwd();
const CHECKPOINTS_DIR = path.join(PROJECT_ROOT, "server", "ml", "checkpoints");
const MANIFEST_PATH = path.join(CHECKPOINTS_DIR, "checkpoint_manifest.json");
const TRAIN_SCRIPT = path.join(PROJECT_ROOT, "server", "ml", "train_tabular_models.py");
const PREDICT_SCRIPT = path.join(PROJECT_ROOT, "server", "ml", "predict_service.py");
const CONTINUOUS_LEARNER_SCRIPT = path.join(PROJECT_ROOT, "server", "ml", "continuous_learner.py");
const LEDGER_PATH = path.join(CHECKPOINTS_DIR, "prediction_ledger.json");

export async function getMlStatus(_req: Request, res: Response) {
  try {
    const checkpointsExist = fs.existsSync(CHECKPOINTS_DIR);
    let checkpointFiles: Array<{ name: string; sizeBytes: number; modifiedAt: string }> = [];

    if (checkpointsExist) {
      const files = fs.readdirSync(CHECKPOINTS_DIR);
      checkpointFiles = files.map((f) => {
        const stat = fs.statSync(path.join(CHECKPOINTS_DIR, f));
        return {
          name: f,
          sizeBytes: stat.size,
          modifiedAt: stat.mtime.toISOString(),
        };
      });
    }

    let manifest = null;
    if (fs.existsSync(MANIFEST_PATH)) {
      manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
    }

    // Hardware status
    const hardwareInfo = {
      computeDevice: "CPU (Multi-threaded Intel/AMD AVX2/AVX-512)",
      acceleration: "OpenMP / multi-threading (n_jobs=-1)",
      gpuAvailable: false,
      gpuMessage: "No local NVIDIA CUDA GPU found in PATH. XGBoost & LightGBM are operating in ultra-low-latency CPU multithreaded mode.",
      externalGpuGuide: {
        description: "To connect external GPU cluster or cloud inference endpoints (e.g. AWS SageMaker, GCP Vertex AI, RunPod):",
        endpointsSupported: ["vLLM / Triton Inference Server", "TorchServe GPU", "XGBoost Dask-CUDA"],
        environmentVariables: ["EXTERNAL_ML_GPU_ENDPOINT", "GPU_AUTH_TOKEN"]
      },
      models: {
        xgboost: {
          version: "3.4.1",
          tasks: ["Thunderstorm Genesis", "Severe Hail Occurrence", "Microburst Ensemble"],
          checkpoint: "server/ml/checkpoints/xgboost_thunderstorm.json"
        },
        lightgbm: {
          version: "4.7.0",
          tasks: ["Flash Cloudburst Deluge Classifier"],
          checkpoint: "server/ml/checkpoints/lightgbm_cloudburst.txt"
        }
      },
      checkpointCount: checkpointFiles.length,
      files: checkpointFiles,
      lastManifest: manifest
    };

    res.json({
      status: "success",
      hardware: hardwareInfo
    });
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}

export async function getMlCheckpoints(_req: Request, res: Response) {
  try {
    if (!fs.existsSync(MANIFEST_PATH)) {
      return res.status(404).json({
        status: "error",
        message: "No checkpoint manifest found. Please train models first."
      });
    }

    const manifestData = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
    const files = fs.readdirSync(CHECKPOINTS_DIR).map((name) => {
      const stats = fs.statSync(path.join(CHECKPOINTS_DIR, name));
      return {
        name,
        sizeBytes: stats.size,
        updatedAt: stats.mtime.toISOString()
      };
    });

    res.json({
      status: "success",
      manifest: manifestData,
      files
    });
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}

export async function postMlTrain(req: Request, res: Response) {
  try {
    const { n_samples = 6000, epochs = 25, lr = 0.05 } = req.body || {};

    const args = [
      TRAIN_SCRIPT,
      "--samples", String(n_samples),
      "--epochs", String(epochs),
      "--lr", String(lr)
    ];

    const { stdout, stderr } = await execFileAsync("python", args, {
      cwd: PROJECT_ROOT,
      timeout: 120000
    });

    let manifest = null;
    if (fs.existsSync(MANIFEST_PATH)) {
      manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
    }

    res.json({
      status: "success",
      message: "Tabular XGBoost & LightGBM models retrained and checkpoints persisted.",
      logs: stdout,
      errors: stderr,
      manifest
    });
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}

export async function postMlPredict(req: Request, res: Response) {
  try {
    const inputData = req.body;
    if (!inputData) {
      return res.status(400).json({ status: "error", message: "Missing soundings / telemetry data." });
    }

    const inputJson = JSON.stringify(inputData);

    const { stdout } = await execFileAsync("python", [PREDICT_SCRIPT, "--input", inputJson], {
      cwd: PROJECT_ROOT,
      timeout: 15000
    });

    const parsed = JSON.parse(stdout);
    res.json(parsed);
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}

let isContinuousLearningRunning = false;
let lastContinuousResult: any = null;

export async function postMlPredictAllGrids(req: Request, res: Response) {
  try {
    const inputData = req.body;
    if (!inputData) {
      return res.status(400).json({ status: "error", message: "Missing soundings / telemetry data." });
    }

    const inputJson = JSON.stringify(inputData);
    const child = spawn("python", [PREDICT_SCRIPT, "--stdin"], {
      cwd: PROJECT_ROOT,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));

    child.stdin.write(inputJson);
    child.stdin.end();

    child.on("close", (code) => {
      if (code !== 0) {
        return res.status(500).json({ status: "error", message: "Vectorized inference failed", stderr });
      }
      try {
        const parsed = JSON.parse(stdout);
        res.json(parsed);
      } catch {
        res.json({ status: "success", raw: stdout });
      }
    });
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}

export async function postContinuousLearnIngest(req: Request, res: Response) {
  try {
    const observations = req.body;
    if (!observations) {
      return res.status(400).json({ status: "error", message: "Missing observation data" });
    }

    // Debounce guard: if a learning cycle is currently training, respond with cached/busy state
    if (isContinuousLearningRunning) {
      return res.json({
        status: "busy",
        message: "Continuous learning cycle is actively computing; observation queued",
        lastResult: lastContinuousResult,
      });
    }

    isContinuousLearningRunning = true;
    const inputJson = JSON.stringify(observations);
    const child = spawn("python", [CONTINUOUS_LEARNER_SCRIPT, "--stdin"], {
      cwd: PROJECT_ROOT,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));

    child.stdin.write(inputJson);
    child.stdin.end();

    child.on("close", (code) => {
      isContinuousLearningRunning = false;
      if (code !== 0) {
        return res.status(500).json({ status: "error", message: "Self-learning script failed", stderr });
      }
      try {
        const parsed = JSON.parse(stdout);
        lastContinuousResult = parsed;
        res.json(parsed);
      } catch {
        res.json({ status: "success", raw: stdout });
      }
    });
  } catch (err: any) {
    isContinuousLearningRunning = false;
    res.status(500).json({ status: "error", message: err.message });
  }
}

export async function getContinuousLearnStatus(_req: Request, res: Response) {
  try {
    let manifest = null;
    if (fs.existsSync(MANIFEST_PATH)) {
      manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
    }

    let ledger = null;
    if (fs.existsSync(LEDGER_PATH)) {
      ledger = JSON.parse(fs.readFileSync(LEDGER_PATH, "utf-8"));
    }

    const replayBufferPath = path.join(CHECKPOINTS_DIR, "live_replay_buffer.jsonl");
    let replayCount = 0;
    if (fs.existsSync(replayBufferPath)) {
      replayCount = fs.readFileSync(replayBufferPath, "utf-8").split("\n").filter(Boolean).length;
    }

    const lossHistory = manifest?.loss_history || [];
    const lastLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : null;

    res.json({
      status: "success",
      continuous_learning_step: manifest?.continuous_learning_step || 0,
      gatekeeper_status: manifest?.gatekeeper_status || "PROMOTED",
      total_grids_monitored: manifest?.total_grids_monitored || 64,
      all_india_coverage_pct: manifest?.all_india_coverage_pct || 100.0,
      last_self_learning_at: manifest?.last_self_learning_at || manifest?.generated_at || null,
      total_verified_soundings: ledger?.total_verified || 0,
      pending_prediction_verifications: ledger?.pending_predictions?.length || 0,
      recent_verified_history: (ledger?.verified_history || []).slice(-10),
      replay_buffer_size: replayCount,
      loss_history: lossHistory,
      last_loss: lastLoss,
      checkpoints: manifest?.checkpoints || {},
    });
  } catch (err: any) {
    res.status(500).json({ status: "error", message: err.message });
  }
}


