import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import {
  getMlStatus,
  getMlCheckpoints,
  postMlTrain,
  postMlPredict,
  postContinuousLearnIngest,
  getContinuousLearnStatus,
} from "./mlController.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const server = createServer(app);

  app.use(express.json());

  // ML Intelligence Endpoints
  app.get("/api/ml/status", getMlStatus);
  app.get("/api/ml/checkpoints", getMlCheckpoints);
  app.post("/api/ml/train", postMlTrain);
  app.post("/api/ml/predict", postMlPredict);
  app.post("/api/ml/continuous-learn/ingest", postContinuousLearnIngest);
  app.get("/api/ml/continuous-learn/status", getContinuousLearnStatus);

  // Serve static files from dist/public in production
  const staticPath =
    process.env.NODE_ENV === "production"
      ? path.resolve(__dirname, "public")
      : path.resolve(__dirname, "..", "dist", "public");

  app.use(express.static(staticPath));

  // Handle client-side routing - serve index.html for all routes
  app.get("*", (_req, res) => {
    res.sendFile(path.join(staticPath, "index.html"));
  });

  const port = process.env.PORT || 3000;

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
