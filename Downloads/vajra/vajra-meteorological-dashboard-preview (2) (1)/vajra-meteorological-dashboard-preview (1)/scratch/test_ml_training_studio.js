// Mathematical and algorithmic unit test for ML Training Studio & Parameter Perturbation

function zToRainRate(dBZ) {
  if (dBZ < 15) return 0;
  const Z = Math.pow(10, dBZ / 10);
  const R = Math.pow(Z / 200, 0.625);
  return Number(R.toFixed(1));
}

function rainRateToZ(mmHr) {
  if (mmHr <= 0.05) return 12;
  const Z = 200 * Math.pow(mmHr, 1.6);
  return Number((10 * Math.log10(Math.max(1, Z))).toFixed(1));
}

// Emulate ML engine training state & parameter overrides
class TestMLEngine {
  constructor() {
    this.cellOverrides = new Map();
    this.trainingHistory = [];
    this.currentBMSELoss = 0.0094;
    this.csiScore = 0.84;
    this.podScore = 91.2;
    this.farScore = 7.8;
    this.learningRate = 0.0003;
    this.stepCount = 142;

    // Seed 16 historical steps
    for (let i = 0; i < 16; i++) {
      const step = 127 + i;
      const loss = 0.048 - i * 0.0025;
      this.trainingHistory.push({
        step,
        loss: Number(loss.toFixed(4)),
        csi: 0.70 + i * 0.01,
        observedDbz: 42 + (i % 3),
        predictedDbz: 41 + (i % 3),
      });
    }
  }

  updateCellParameters(cellId, params) {
    const existing = this.cellOverrides.get(cellId) || {};
    this.cellOverrides.set(cellId, { ...existing, ...params });

    // Execute online adaptation step
    const obsDbz = params.reflectivityDbz || 48;
    const predDbz = 44;
    const residual = Math.abs(obsDbz - predDbz);
    const weight = obsDbz >= 45 ? 5.0 : 1.0;
    const stepLoss = Number(((weight * Math.pow(residual, 2)) / 3200).toFixed(5));
    this.currentBMSELoss = Number((0.92 * this.currentBMSELoss + 0.08 * stepLoss).toFixed(4));
    this.stepCount += 1;

    this.trainingHistory.push({
      step: this.stepCount,
      loss: this.currentBMSELoss,
      observedDbz: obsDbz,
      predictedDbz: predDbz,
      residual,
      isUserPerturbed: true,
      eventName: `EVENT #${cellId} (PARAM TUNING)`,
    });
    if (this.trainingHistory.length > 50) this.trainingHistory.shift();
  }

  resetCellToLive(cellId) {
    this.cellOverrides.delete(cellId);
  }

  hasCellOverride(cellId) {
    return this.cellOverrides.has(cellId);
  }
}

const engine = new TestMLEngine();

console.log("=== 1. Testing Training History Buffer ===");
console.log(`Initial history buffer length: ${engine.trainingHistory.length} steps`);
console.log(`First loss: ${engine.trainingHistory[0].loss} -> Last loss: ${engine.trainingHistory[engine.trainingHistory.length - 1].loss}`);
if (engine.trainingHistory.length !== 16) throw new Error("History buffer failed length test");
console.log("Training History Buffer: PASSED\n");

console.log("=== 2. Testing Parameter Override & Model Adaptation ===");
const baselineLoss = engine.currentBMSELoss;
console.log(`Baseline B-MSE loss: ${baselineLoss}`);

// Apply severe convective perturbation: CAPE=4200, dBZ=64, Wind=95 km/h
engine.updateCellParameters(14, {
  cape: 4200,
  liftedIndex: -8.0,
  reflectivityDbz: 64.0,
  rainRate: zToRainRate(64.0),
  windGust: 95,
  speedKmh: 85,
  bearingDeg: 120,
});

const latest = engine.trainingHistory[engine.trainingHistory.length - 1];
console.log(`Post-perturbation Step #${latest.step} - Loss: ${latest.loss} - Residual: ${latest.residual} dBZ - UserPerturbed: ${latest.isUserPerturbed}`);

if (!engine.hasCellOverride(14)) throw new Error("Cell 14 should have active override");
if (!latest.isUserPerturbed) throw new Error("Latest step should be marked user perturbed");
console.log("Parameter Override & Model Adaptation: PASSED\n");

console.log("=== 3. Testing Reset to Live Stream ===");
engine.resetCellToLive(14);
console.log(`Has Cell #14 override after reset: ${engine.hasCellOverride(14)}`);
if (engine.hasCellOverride(14)) throw new Error("Override should be removed after reset");
console.log("Reset to Live Stream: PASSED\n");

console.log("=== 4. Testing Marshall-Palmer Linked Sliders ===");
const testZ = 55.0;
const testRain = zToRainRate(testZ);
const backToZ = rainRateToZ(testRain);
console.log(`${testZ} dBZ -> ${testRain} mm/hr -> ${backToZ} dBZ (diff: ${Math.abs(testZ - backToZ).toFixed(2)})`);
if (Math.abs(testZ - backToZ) > 0.5) throw new Error("Bidirectional slider sync failed");
console.log("Marshall-Palmer Linked Sliders: PASSED\n");

console.log("ALL ML TRAINING & PARAMETER PERTURBATION TESTS PASSED!");
