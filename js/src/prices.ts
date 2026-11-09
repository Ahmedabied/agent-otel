// Token prices, supplied by you, in the same table format as the Python side:
//   {"currency": "USD", "models": {"my-model-*": {"input": 0.5, "output": 1.5, "cache_read": 0.05}}}
// Prices are per million tokens. The first matching entry wins, in the order written. Globs work like Python's
// fnmatch: * any run of characters, ? one character, [abc] and [!abc] one character from (or not from) a set.

export interface Price {
  input: number;
  output: number;
  cache_read?: number | null;
  cache_write?: number | null;
}

export interface PriceTable {
  currency?: string;
  models: Record<string, Price>;
}

export class PriceError extends Error {}

function globToRegExp(glob: string): RegExp {
  let out = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i]!;
    if (c === "*") out += "[\\s\\S]*";
    else if (c === "?") out += "[\\s\\S]";
    else if (c === "[") {
      const end = glob.indexOf("]", i + 2);
      if (end < 0) {
        out += "\\[";
        continue;
      }
      let set = glob.slice(i + 1, end);
      const negate = set.startsWith("!");
      if (negate) set = set.slice(1);
      out += `[${negate ? "^" : ""}${set.replace(/[\\\]^]/g, "\\$&")}]`;
      i = end;
    } else out += c.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
  }
  return new RegExp(`^${out}$`);
}

export class Prices {
  readonly currency: string;
  private readonly entries: Array<[RegExp, Price]> = [];

  constructor(table: PriceTable) {
    if (!table || typeof table !== "object" || !table.models || typeof table.models !== "object") {
      throw new PriceError('a price table is {"currency": "...", "models": {"<model glob>": {"input": x, "output": y}}}');
    }
    this.currency = String(table.currency ?? "USD");
    for (const [pattern, p] of Object.entries(table.models)) {
      if (!p || typeof p !== "object" || !("input" in p) || !("output" in p)) {
        throw new PriceError(`price for '${pattern}' needs input and output (per million tokens)`);
      }
      const nums = [p.input, p.output, p.cache_read ?? 0, p.cache_write ?? 0];
      if (nums.some((n) => typeof n !== "number" || !Number.isFinite(n))) {
        throw new PriceError(`price for '${pattern}' must be numbers`);
      }
      if (nums.some((n) => n < 0)) throw new PriceError(`price for '${pattern}' cannot be negative`);
      this.entries.push([globToRegExp(pattern), p]);
    }
  }

  lookup(model: string): Price | null {
    for (const [re, price] of this.entries) if (re.test(model)) return price;
    return null;
  }

  /**
   * Cost of one call, or null when the model is not in the table. inputTokens includes the cached ones (the
   * OpenTelemetry GenAI meaning); the cache counts say how many of them were read from or written to the cache.
   */
  cost(model: string, inputTokens = 0, outputTokens = 0, cacheReadTokens = 0, cacheWriteTokens = 0): number | null {
    const p = this.lookup(model);
    if (!p) return null;
    const read = Math.min(cacheReadTokens, inputTokens);
    const write = Math.min(cacheWriteTokens, inputTokens - read);
    const fresh = inputTokens - read - write;
    const rateRead = p.cache_read ?? p.input;
    const rateWrite = p.cache_write ?? p.input;
    const total = fresh * p.input + read * rateRead + write * rateWrite + outputTokens * p.output;
    return Math.round((total / 1_000_000) * 1e10) / 1e10;
  }
}
