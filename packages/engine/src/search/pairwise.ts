// Ports numpy's pairwise summation (docs/design/clientside-engine-port.md
// section 4.3 M1/M2's reduction table): n < 8 a plain loop; n <= 128 eight
// strided accumulators combined ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) plus a
// tail loop; larger n recursive halving at a multiple of 8. This is
// numpy's `pairwise_sum` (numpy/core/src/umath/loops_utils.h.src),
// PW_BLOCKSIZE = 128.

const BLOCKSIZE = 128;

function sumF64(values: Float64Array, offset: number, n: number): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += values[offset + i]!;
    return res;
  }
  if (n <= BLOCKSIZE) {
    const r = [
      values[offset + 0], values[offset + 1], values[offset + 2], values[offset + 3],
      values[offset + 4], values[offset + 5], values[offset + 6], values[offset + 7],
    ];
    let i = 8;
    const limit = n - (n % 8);
    for (; i < limit; i += 8) {
      for (let j = 0; j < 8; j++) r[j]! += values[offset + i + j]!;
    }
    let res = ((r[0]! + r[1]!) + (r[2]! + r[3]!)) + ((r[4]! + r[5]!) + (r[6]! + r[7]!));
    for (; i < n; i++) res += values[offset + i]!;
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return sumF64(values, offset, n2) + sumF64(values, offset + n2, n - n2);
}

// Same tree as sumF64, but every intermediate addition is rounded to
// float32: storing into a Float32Array rounds on assignment exactly like
// Math.fround, so only the final 7-term combine (plain numbers, not a
// typed-array store) needs an explicit fround per step.
function sumF32(values: Float32Array, offset: number, n: number): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res = Math.fround(res + values[offset + i]!);
    return res;
  }
  if (n <= BLOCKSIZE) {
    const r = new Float32Array(8);
    for (let j = 0; j < 8; j++) r[j] = values[offset + j]!;
    let i = 8;
    const limit = n - (n % 8);
    for (; i < limit; i += 8) {
      for (let j = 0; j < 8; j++) r[j] = r[j]! + values[offset + i + j]!;
    }
    const ab = Math.fround(Math.fround(r[0]! + r[1]!) + Math.fround(r[2]! + r[3]!));
    const cd = Math.fround(Math.fround(r[4]! + r[5]!) + Math.fround(r[6]! + r[7]!));
    let res = Math.fround(ab + cd);
    for (; i < n; i++) res = Math.fround(res + values[offset + i]!);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return Math.fround(sumF32(values, offset, n2) + sumF32(values, offset + n2, n - n2));
}

export function pairwiseSum(values: Float64Array): number {
  return sumF64(values, 0, values.length);
}

export function pairwiseSumF32(values: Float32Array): number {
  return sumF32(values, 0, values.length);
}

// subtree_wdl_sum = wdl_sum.sum(axis=0) (node.py:403-405): a C-order (L,3)
// array reduced along axis 0 is numpy's plain row-sequential loop per
// column, not the contiguous pairwise path (section 4.3's reduction table).
export function sumRowsSequential(flat: Float64Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(cols);
  for (let i = 0; i < rows; i++) {
    const base = i * cols;
    for (let j = 0; j < cols; j++) out[j]! += flat[base + j]!;
  }
  return out;
}
