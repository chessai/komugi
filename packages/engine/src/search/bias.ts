// Ports src/komugi/search/match/bias.py: design §A2's subtree-value-bias
// table, key -> (err_sum, weight_sum, gen) (docs/design/clientside-engine-port.md
// section 2.6).

export const WDL = 3;

export type BiasEntry = {
  errSum: Float64Array;
  weightSum: number;
  gen: number;
};

function freshEntry(): BiasEntry {
  return { errSum: new Float64Array(WDL), weightSum: 0.0, gen: 0 };
}

// SubtreeValueBias (bias.py:120-307). `entries` is public, not hidden behind
// an accessor: this port has no private-field convention elsewhere (Board's
// fields are all public too), and tests reconstructing a table from a
// recorded fixture (which has no equivalent of a from-scratch `update` replay)
// need to seed entries directly.
export class SubtreeValueBias {
  freeProp: number;
  weightFloor: number;
  entries: Map<number, BiasEntry>;

  constructor(opts: { freeProp: number; weightFloor: number }) {
    this.freeProp = opts.freeProp;
    this.weightFloor = opts.weightFloor;
    this.entries = new Map();
  }

  keys(): number[] {
    return Array.from(this.entries.keys()).sort((a, b) => a - b);
  }

  entry(key: number): BiasEntry | undefined {
    return this.entries.get(key);
  }

  generation(key: number): number {
    let stored = this.entries.get(key);
    if (stored === undefined) {
      stored = freshEntry();
      this.entries.set(key, stored);
    }
    return stored.gen;
  }

  // read (bias.py:174-190): the mean WDL error under `key`, or zeros below
  // weightFloor. This is the only place weightFloor is applied; storage
  // keeps a legitimately small positive weight_sum verbatim.
  read(key: number): Float64Array {
    const stored = this.entries.get(key);
    if (stored === undefined || stored.weightSum < this.weightFloor) {
      return new Float64Array(WDL);
    }
    const out = new Float64Array(WDL);
    for (let i = 0; i < WDL; i++) out[i] = stored.errSum[i]! / stored.weightSum;
    return out;
  }

  // update (bias.py:192-278): replace one node's contribution to `key`,
  // scaling the removal by free_prop ** (entry.gen - node.gen) so a removal
  // never takes out mass the entry has since decayed past (design §A2).
  update(opts: {
    key: number;
    contribution: Float64Array;
    weight: number;
    gen: number;
    delta: Float64Array;
    newWeight: number;
  }): { contribution: Float64Array; weight: number; gen: number } {
    const { key, contribution, weight, gen, delta, newWeight } = opts;
    for (let i = 0; i < WDL; i++) {
      if (!Number.isFinite(delta[i])) {
        throw new Error(
          `a non-finite subtree-value-bias update for key ${key}: delta=${Array.from(delta)}, ` +
            `new_weight=${newWeight}`,
        );
      }
    }
    if (!Number.isFinite(newWeight)) {
      throw new Error(`a non-finite subtree-value-bias update for key ${key}: new_weight=${newWeight}`);
    }
    let stored = this.entries.get(key);
    if (stored === undefined) {
      stored = freshEntry();
      this.entries.set(key, stored);
    }

    const scale = Math.pow(this.freeProp, stored.gen - gen);
    for (let i = 0; i < WDL; i++) stored.errSum[i]! -= scale * contribution[i]!;
    stored.weightSum -= scale * weight;
    for (let i = 0; i < WDL; i++) stored.errSum[i]! *= this.freeProp;
    stored.weightSum *= this.freeProp;
    stored.gen += 1;

    if (stored.weightSum < -1e-9) {
      throw new Error(
        `subtree-value-bias entry ${key} reached weight_sum=${stored.weightSum} after removing a ` +
          `contribution written at generation ${gen} from an entry now at generation ${stored.gen}`,
      );
    }
    if (stored.weightSum < 0.0) stored.weightSum = 0.0;

    const newContribution = new Float64Array(WDL);
    for (let i = 0; i < WDL; i++) newContribution[i] = newWeight * delta[i]!;
    for (let i = 0; i < WDL; i++) stored.errSum[i]! += newContribution[i]!;
    stored.weightSum += newWeight;
    return { contribution: newContribution, weight: newWeight, gen: stored.gen };
  }

  // decay_all (bias.py:280-299): design §A4's rehome decay, applied to every
  // entry including ones with no live node.
  decayAll(): void {
    for (const stored of this.entries.values()) {
      for (let i = 0; i < WDL; i++) stored.errSum[i]! *= this.freeProp;
      stored.weightSum *= this.freeProp;
      stored.gen += 1;
    }
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}
