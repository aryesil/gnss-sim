"""Shared Galileo E1 I/NAV message field layouts (Galileo OS SIS ICD Issue 2.0).

Word types 0, 1, 2, 3, 4, 5.
"""
from __future__ import annotations

_2 = 2.0

PAGE_SYMS = 500       # 2 s nominal page @ 250 sym/s
HALF_PAGE_SYMS = 250  # 1 s page part (even or odd)
SYNC_SYMS = 10        # 10-symbol sync pattern
CODED_SYMS = 240      # 240 symbols per page part (30x8 interleaver)
DATA_K_BITS = 112     # even page part data field
DATA_J_BITS = 16      # odd page part data field
WORD_BITS = 128       # 112 + 16
CRC_BITS = 24

SYNC_PATTERN = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]

LAYOUT: dict[int, list[tuple[str, str, float, int]]] = {
    1: [
        ("type",      "u", 1.0,        6),
        ("iodnav",    "u", 1.0,       10),
        ("toe",       "u", 60.0,      14),
        ("m0",        "s", _2 ** -31, 32),
        ("e",         "u", _2 ** -33, 32),
        ("sqrtA",     "u", _2 ** -19, 32),
        ("_reserved", "u", 0.0,        2),
    ],
    2: [
        ("type",      "u", 1.0,        6),
        ("iodnav",    "u", 1.0,       10),
        ("omega0",    "s", _2 ** -31, 32),
        ("i0",        "s", _2 ** -31, 32),
        ("omega",     "s", _2 ** -31, 32),
        ("idot",      "s", _2 ** -43, 14),
        ("_reserved", "u", 0.0,        2),
    ],
    3: [
        ("type",      "u", 1.0,        6),
        ("iodnav",    "u", 1.0,       10),
        ("omega_dot", "s", _2 ** -43, 24),
        ("delta_n",   "s", _2 ** -43, 16),
        ("cuc",       "s", _2 ** -29, 16),
        ("cus",       "s", _2 ** -29, 16),
        ("crc",       "s", _2 ** -5,  16),
        ("crs",       "s", _2 ** -5,  16),
        ("sisa",      "u", 1.0,        8),
    ],
    4: [
        ("type",      "u", 1.0,        6),
        ("iodnav",    "u", 1.0,       10),
        ("prn",       "u", 1.0,        6),
        ("cic",       "s", _2 ** -29, 16),
        ("cis",       "s", _2 ** -29, 16),
        ("toc",       "u", 60.0,      14),
        ("af0",       "s", _2 ** -34, 31),
        ("af1",       "s", _2 ** -46, 21),
        ("af2",       "s", _2 ** -59,  6),
        ("_reserved", "u", 0.0,        2),
    ],
    5: [
        ("type",        "u", 1.0,        6),
        ("ai0",         "u", 1.0,       11),
        ("ai1",         "u", 1.0,       11),
        ("ai2",         "u", 1.0,       14),
        ("region",      "u", 1.0,        5),
        ("bgd_e1_e5a",  "s", _2 ** -32, 10),
        ("bgd_e1_e5b",  "s", _2 ** -32, 10),
        ("_reserved",   "u", 0.0,        6),
        ("week",        "u", 1.0,       12),
        ("tow",         "u", 1.0,       20),
        ("_reserved",   "u", 0.0,       23),
    ],
    0: [
        ("type",        "u", 1.0,        6),
        ("time_valid",  "u", 1.0,        2),
        ("_reserved",   "u", 0.0,       88),
        ("week",        "u", 1.0,       12),
        ("tow",         "u", 1.0,       20),
    ],
}
