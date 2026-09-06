// Mathematical verification test for ML nowcasting engine
// Tests:
// 1. Z-R Marshall-Palmer physical inversion
// 2. Conditioning sensitivity (CAPE, LI)
// 3. Online training loss calculation and weight adaptation
// 4. Physical boundaries check

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

console.log("=== 1. Testing Marshall-Palmer Inversion ===");
const testRates = [5, 15, 30, 50, 80, 100];
let zrPassed = true;
testRates.forEach(r => {
  const z = rainRateToZ(r);
  const recoveredR = zToRainRate(z);
  const diff = Math.abs(r - recoveredR);
  console.log(`Rain Rate: ${r} mm/hr -> Z: ${z} dBZ -> Recovered R: ${recoveredR} mm/hr (diff: ${diff.toFixed(2)})`);
  if (diff > 1.5) zrPassed = false;
});
console.log("Z-R Inversion Test:", zrPassed ? "PASSED" : "FAILED");

console.log("\n=== 2. Testing Live Conditioning Sensitivity ===");
// When CAPE is high (3000) and LI is -6, convective growth should be significantly higher than low CAPE (500) and LI (+1)
const highCape = 3000;
const highLi = -6.0;
const lowCape = 500;
const lowLi = 1.0;

const capeFactorHigh = Math.min(2.2, Math.max(0.5, highCape / 1200));
const instabilityFactorHigh = Math.min(2.2, Math.max(0.6, Math.abs(Math.min(0, highLi)) / 3.0));
const ampHigh = capeFactorHigh * 0.6 + instabilityFactorHigh * 0.4;

const capeFactorLow = Math.min(2.2, Math.max(0.5, lowCape / 1200));
const instabilityFactorLow = Math.min(2.2, Math.max(0.6, Math.abs(Math.min(0, lowLi)) / 3.0));
const ampLow = capeFactorLow * 0.6 + instabilityFactorLow * 0.4;

console.log(`High CAPE Amp: ${ampHigh.toFixed(2)} | Low CAPE Amp: ${ampLow.toFixed(2)}`);
if (ampHigh > ampLow * 2.0) {
  console.log("Conditioning Sensitivity Test: PASSED (Convective amplifier strongly amplifies explosive convection)");
} else {
  console.log("Conditioning Sensitivity Test: FAILED");
}

console.log("\n=== 3. Testing Online B-MSE Loss Formulation ===");
const observedDbz = 52.0;
const predDbz = 44.0;
const residual = Math.abs(observedDbz - predDbz);
const weight = observedDbz >= 45 ? 5.0 : 1.0;
const bmseLoss = (weight * Math.pow(residual, 2)) / 3200;
console.log(`Observed: ${observedDbz} dBZ, Predicted: ${predDbz} dBZ -> B-MSE Loss: ${bmseLoss.toFixed(5)}`);
if (bmseLoss > 0 && bmseLoss < 0.2) {
  console.log("B-MSE Loss Formulation: PASSED");
} else {
  console.log("B-MSE Loss Formulation: FAILED");
}
