// Engine-wide constants (src/komugi/engine/cyboard.pyx:55-104, cymovegen.pyx:46-69, fen.py:20-37).

// Piece type indices, in cyboard's canonical order. Piece code = ti*2 + color + 1; 0 = empty.
export const MARSHAL = 0;
export const GENERAL = 1;
export const LIEUTENANT_GENERAL = 2;
export const MAJOR_GENERAL = 3;
export const WARRIOR = 4;
export const LANCER = 5;
export const RIDER = 6;
export const SPY = 7;
export const FORTRESS = 8;
export const SOLDIER = 9;
export const CANNON = 10;
export const ARCHER = 11;
export const MUSKETEER = 12;
export const TACTICIAN = 13;
export const NUM_PIECE_TYPES = 14;

export const WHITE = 0;
export const BLACK = 1;

// Kanji per type index (cyboard.PIECE_TYPES).
export const PIECE_KANJI: readonly string[] = [
  '帥', '大', '中', '小', '侍', '槍', '馬',
  '忍', '砦', '兵', '砲', '弓', '筒', '謀',
];

// Lowercase FEN letter per type index (cyboard._IDX_TO_FEN_LETTER).
export const FEN_LETTERS: readonly string[] = [
  'm', 'g', 'i', 'j', 'w', 'n', 'r', 's', 'f', 'd', 'c', 'a', 'k', 't',
];

// Lowercase FEN letter -> type index (cyboard._FEN_LETTER_TO_IDX).
export const FEN_LETTER_TO_INDEX: Readonly<Record<string, number>> = Object.freeze(
  Object.fromEntries(FEN_LETTERS.map((l, i) => [l, i])),
);

// Setup modes (utils.setup_mode_to_code).
export const MODE_INTRO = 0;
export const MODE_BEGINNER = 1;
export const MODE_INTERMEDIATE = 2;
export const MODE_ADVANCED = 3;

// Board geometry: cells[(r*9 + f)*3 + t], r = rank-1, f = file-1, t = tier-1.
export const RANKS = 9;
export const FILES = 9;
export const TIERS = 3;
export const NUM_SQUARES = 81;
export const NUM_CELLS = 243;
export const HAND_SIZE = NUM_PIECE_TYPES * 2;

// Move types (cymovegen._MT_*).
export const MT_ROUTE = 0;
export const MT_TSUKE = 1;
export const MT_CAPTURE = 2;
export const MT_BETRAY = 3;
export const MT_ARATA = 4;

// Action space (cymovegen.pyx:65-69, action_encoding.py:56-63,88).
export const A0_START = 0;
export const A1_START = 6561;
export const B_START = 32805;
export const C_START = 35073;
export const ACTION_SIZE = 37341;

// Versions the 37341-id space itself (action_encoding.py's ACTION_ENCODING_VERSION twin).
export const ACTION_ENCODING_VERSION = 2;

export const INTRO_POSITION =
  '3img3/1s2n2s1/d1fwdwf1d/9/9/9/D1FWDWF1D/1S2N2S1/3GMI3 J2N2R2D1/j2n2r2d1 w 0 - 1';
export const BEGINNER_POSITION =
  '3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1FWDWF1D/1SA1N1AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 w 1 - 1';
export const INTERMEDIATE_POSITION =
  '9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 2 wb 1';
export const ADVANCED_POSITION =
  '9/9/9/9/9/9/9/9/9 M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1';
