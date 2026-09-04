import fs from 'node:fs';
const file = 'client/src/pages/Home.tsx';
let s = fs.readFileSync(file, 'utf8');
s = s.replace('</div>}</section>\n\n      <footer', '</div></div>}</section>\n\n      <footer');
fs.writeFileSync(file, s);
