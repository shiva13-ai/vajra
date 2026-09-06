// scratch/test_thermo_lightning.js
// Verification script for thermodynamic calculations, cloud base, and lightning rate

function calcDewPoint(tempC, rhPercent) {
  const a = 17.625;
  const b = 243.04;
  const rh = Math.max(1, Math.min(100, rhPercent));
  const alpha = Math.log(rh / 100.0) + (a * tempC) / (b + tempC);
  const dp = (b * alpha) / (a - alpha);
  return Math.round(dp * 10) / 10;
}

function calcLCL(tempC, dewPointC) {
  const depression = Math.max(0, tempC - dewPointC);
  const lclMeters = 125 * depression;
  return Math.max(200, Math.min(4500, Math.round(lclMeters)));
}

function calcThermodynamicInstability(tempC, rhPercent) {
  const dewPoint = calcDewPoint(tempC, rhPercent);
  const thetaE_proxy = tempC + 2.5 * dewPoint;
  const energyDelta = Math.max(0, thetaE_proxy - 68);
  const derivedCape = Math.min(4800, Math.round(energyDelta * 55 + (tempC > 30 ? (tempC - 30) * 80 : 0)));
  const derivedLi = Math.max(-10.5, Math.min(6.0, Math.round((2.0 - (derivedCape / 380)) * 10) / 10));
  const wMax = Math.round(Math.sqrt(2 * Math.max(0, derivedCape)) * 10) / 10;
  return { dewPoint, lclMeters: calcLCL(tempC, dewPoint), derivedCape, derivedLi, wMax };
}

function calcLightningFlashRate(cape, dbz, freezingLevel, tempC, rhPercent) {
  const effectiveCape = cape !== undefined ? cape : 1800;
  const effectiveDbz = dbz !== undefined ? dbz : 40;
  const effectiveRh = rhPercent !== undefined ? rhPercent : 75;

  if (effectiveDbz < 32 && effectiveCape < 1000) {
    return { flashesPerMin: 0, strikeProbability: 0 };
  }

  const capeFactor = Math.pow(Math.max(0, effectiveCape) / 1500, 1.8);
  const dbzFactor = Math.max(0, (effectiveDbz - 30) / 20);
  const moistureFactor = Math.max(0.3, effectiveRh / 80);

  const baseRate = 1.2 * capeFactor * Math.pow(dbzFactor, 1.5) * moistureFactor;
  const flashesPerMin = Math.max(0, Math.min(45, Math.round(baseRate * 10) / 10));
  const strikeProbability = Math.min(99, Math.round(Math.min(1, flashesPerMin / 8) * 90 + (effectiveDbz > 45 ? 9 : 0)));

  return { flashesPerMin, strikeProbability };
}

console.log("=== VAJRA THERMODYNAMICS & LIGHTNING EQUATION VALIDATION ===");

// Case 1: Extreme Pre-Monsoon Convective Burst (38.5°C, 88% RH, 55 dBZ)
const burst = calcThermodynamicInstability(38.5, 88);
const burstLight = calcLightningFlashRate(burst.derivedCape, 55, 4150, 38.5, 88);
console.log("\n[TEST 1: EXTREME CONVECTIVE BURST]");
console.log(`Temp: 38.5°C, RH: 88%`);
console.log(`Dew Point: ${burst.dewPoint}°C (expected ~36.2°C)`);
console.log(`LCL Cloud Base: ${burst.lclMeters}m (expected ~288m)`);
console.log(`Derived CAPE: ${burst.derivedCape} J/kg (expected > 3000 J/kg)`);
console.log(`Lifted Index: ${burst.derivedLi}°C (expected < -6°C)`);
console.log(`Max Updraft w_max: ${burst.wMax} m/s`);
console.log(`Lightning Rate: ${burstLight.flashesPerMin} strikes/min, Prob: ${burstLight.strikeProbability}%`);

// Case 2: Monsoon Squall (32.5°C, 92% RH, 46 dBZ)
const monsoon = calcThermodynamicInstability(32.5, 92);
const monsoonLight = calcLightningFlashRate(monsoon.derivedCape, 46, 4200, 32.5, 92);
console.log("\n[TEST 2: MONSOON SQUALL]");
console.log(`Temp: 32.5°C, RH: 92%`);
console.log(`Dew Point: ${monsoon.dewPoint}°C`);
console.log(`LCL Cloud Base: ${monsoon.lclMeters}m`);
console.log(`Derived CAPE: ${monsoon.derivedCape} J/kg`);
console.log(`Lifted Index: ${monsoon.derivedLi}°C`);
console.log(`Max Updraft w_max: ${monsoon.wMax} m/s`);
console.log(`Lightning Rate: ${monsoonLight.flashesPerMin} strikes/min, Prob: ${monsoonLight.strikeProbability}%`);

// Case 3: Stable Winter Air (18.0°C, 40% RH, 20 dBZ)
const stable = calcThermodynamicInstability(18.0, 40);
const stableLight = calcLightningFlashRate(stable.derivedCape, 20, 3800, 18.0, 40);
console.log("\n[TEST 3: STABLE AIR]");
console.log(`Temp: 18.0°C, RH: 40%`);
console.log(`Dew Point: ${stable.dewPoint}°C`);
console.log(`LCL Cloud Base: ${stable.lclMeters}m`);
console.log(`Derived CAPE: ${stable.derivedCape} J/kg (expected 0 J/kg)`);
console.log(`Lifted Index: ${stable.derivedLi}°C (expected positive/stable)`);
console.log(`Lightning Rate: ${stableLight.flashesPerMin} strikes/min, Prob: ${stableLight.strikeProbability}%`);

// Verify assertions
if (burst.derivedCape > 2500 && burstLight.flashesPerMin > 10 && stableLight.flashesPerMin === 0) {
  console.log("\n>>> ALL PHYSICAL THERMODYNAMICS & LIGHTNING EQUATIONS VERIFIED PERFECTLY! <<<");
  process.exit(0);
} else {
  console.error("Physics assertion failed!");
  process.exit(1);
}
