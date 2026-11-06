// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Check that MathJax can typeset the math in Sphinx HTML output.
//
// Sphinx passes LaTeX through to the page unparsed, and MathJax typesets
// it in the browser, so LaTeX errors do not fail the build. This script
// runs each math node through the MathJax TeX input jax used by the
// pages, with undefined macros made errors, and exits with a non-zero
// status if any node fails.
//
// Requires the MathJax npm package:
//
//     npm install --no-save @mathjax/src@4.1.3
//     node check_math.mjs build/html

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const MathJax = await require("@mathjax/src").init({
  loader: { load: ["input/tex"] },
  tex: {
    packages: { "[-]": ["noundefined"] },
    formatError: (jax, err) => {
      throw err;
    },
  },
});

function htmlFiles(path) {
  if (!statSync(path).isDirectory()) return path.endsWith(".html") ? [path] : [];
  return readdirSync(path).flatMap((f) => htmlFiles(join(path, f)));
}

const unescape = (s) =>
  s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");

// Sphinx writes inline math as \(...\) and display math as \[...\]
const mathNode = /class="math notranslate nohighlight">([\s\S]*?)<\/(span|div)>/g;

let count = 0;
let errors = 0;
for (const file of process.argv.slice(2).flatMap(htmlFiles)) {
  for (const [, body] of readFileSync(file, "utf8").matchAll(mathNode)) {
    count++;
    let tex = unescape(body).trim();
    const display = !tex.startsWith("\\(");
    if (tex.startsWith("\\(") || tex.startsWith("\\[")) tex = tex.slice(2, -2);
    try {
      MathJax.tex2mml(tex, { display });
    } catch (err) {
      errors++;
      console.error(`${file}: ${err.message}\n    ${tex.replace(/\s+/g, " ")}`);
    }
  }
}

console.log(`Checked ${count} math nodes: ${errors} error(s)`);
process.exit(errors > 0 ? 1 : 0);
