import fs from 'node:fs';
const file = 'client/src/pages/Home.tsx';
let s = fs.readFileSync(file, 'utf8');
s = s.replace('{showForecast ? "−" : "+"}</button>{showForecast && <div className="playback-content">', '{showForecast ? "−" : "+ 05 / FORECAST"}</button>{showForecast && <div className="playback-content">');
fs.writeFileSync(file, s);
