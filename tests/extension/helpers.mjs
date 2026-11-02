// Loads the extension's classic scripts into a Node vm context, the way Chrome loads them into a
// page or service worker (no bundler, no npm). Run with `make test-ext`.
import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

export const ROOT = fileURLToPath(new URL("../../", import.meta.url));

export function readRepoFile(path) {
  return readFileSync(ROOT + path, "utf8");
}

/** A fresh context with the web globals the extension uses, plus `globals`. */
export function createContext(globals = {}) {
  const context = vm.createContext({
    URL,
    AbortSignal,
    console,
    setTimeout,
    clearTimeout,
    structuredClone,
    crypto: webcrypto,
    ...globals,
  });
  context.importScripts = (...files) => {
    for (const file of files) runScript(context, `extension/${file}`);
  };
  return context;
}

export function runScript(context, path) {
  vm.runInContext(readRepoFile(path), context, { filename: path });
}

/** Runs extension scripts in order (like <script> tags) and returns the context. */
export function loadExtensionScripts(files, globals = {}) {
  const context = createContext(globals);
  for (const file of files) runScript(context, `extension/${file}`);
  return context;
}

/** Objects made inside the vm have another realm's prototypes; compare them as plain data. */
export function plain(value) {
  return JSON.parse(JSON.stringify(value));
}
