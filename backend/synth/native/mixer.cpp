// backend/synth/native/mixer.cpp
#include "mixer.hpp"
#include "nco.hpp"
#include <climits>
#include <cmath>

#include <vector>

namespace gs {

// CBOC(6,1,1/11) weights: sqrt(10/11), sqrt(1/11).
static constexpr float kCbocA = 0.95346258924559231f;
static constexpr float kCbocB = 0.30151134457776363f;
static constexpr int kMixTile = 2048;
// 4096 girişli LUT'a faz indeksi: üst 12 bit. Kesmek yerine yarım LSB (2^19)
// eklenip yuvarlanır; böylece faz hatası [0, 2π/4096) yerine ±π/4096 olur
// (sabit 0.043° ofset ve faz-kesme spur'ının yarısı gider). uint32 taşması
// 4095 -> 0 sarmasını doğal olarak verir.
static constexpr uint32_t kLutRound = 1u << 19;

struct SvPrepared {
    const SvChannel *sv = nullptr;
    double abs_t0 = 0.0;
    bool use_traj = false;
    double chip_rate = 0.0;
    float g = 1.0f;
    int L = 1023;
    const int8_t *code = nullptr;
    NavSource nav{};
    double eff = 0.0;
    double cp_base = 0.0;
    bool use_boc = false;
    int cboc = 0;
    const int8_t *sec = nullptr;
    int sec_len = 0;
    bool use_tx = false;
    double tx_off = 0.0;
    double chips_per_sym = 0.0;
    bool use_gk = false;
    double inv_gdt = 0.0;
    int64_t gk_last = 0;
    bool cache_chip = false;
    bool has_sec = false;
    bool has_nav = false;
    uint32_t carr_phase0 = 0;
    uint32_t carr_inc = 0;
    // Running cache state across tiles for this SV
    int64_t last_fl = INT64_MIN;
    int64_t last_ufl = INT64_MIN;
    int64_t last_ci = 0;
    int64_t last_period = INT64_MIN;
    int64_t last_si = 0;
    int64_t last_sym_idx = INT64_MIN;
    float last_nav_sym = 1.0f;
    float cached = 0.0f;
    float cached_d = 0.0f;
};

static inline SvPrepared prepare_sv(const SvChannel &sv, double fs,
                                    uint64_t sample0) {
    const double dt = 1.0 / fs;
    SvPrepared p;
    p.sv = &sv;
    p.abs_t0 = sample0 * dt;
    p.use_traj = sv.traj_nknots > 0;
    double eff_carr_freq = sv.carrier_freq_hz;
    double eff_carr_phase0 = sv.carrier_phase0_rad;
    double eff_code_rate = sv.code_rate_hz + sv.code_doppler_hz;
    double seg_t0 = 0.0;
    double code_phase_at_seg = 0.0;
    if (p.use_traj) {
        const uint64_t ks = sv.traj_knot_samples;
        int j = ks ? static_cast<int>(sample0 / ks) : 0;
        if (j < 0) j = 0;
        if (j > sv.traj_nknots - 1) j = sv.traj_nknots - 1;
        seg_t0 = static_cast<double>(static_cast<uint64_t>(j) * ks) / fs;
        eff_carr_freq = sv.traj_carr_freq[j];
        eff_carr_phase0 = sv.traj_carr_phase[j];
        eff_code_rate = sv.traj_code_rate[j];
        code_phase_at_seg = sv.traj_code_phase[j];
    }

    Nco carr;
    carr.set_freq(eff_carr_freq, fs);
    double ph0 = eff_carr_phase0
                 + 2.0 * M_PI * eff_carr_freq * (p.abs_t0 - seg_t0);
    ph0 = std::fmod(ph0, 2.0 * M_PI);
    // Faz çevrim (cycle) kesrine indirilir. Küçük negatif ph0'a +2π eklenince
    // tam 2π'ye yuvarlanıp oran*2^32 == 2^32 olabilir; double -> uint32 dönüşümü
    // o durumda tanımsız davranış (UB) olurdu. uint64 üzerinden 32 bite sarılır.
    double cyc = ph0 / (2.0 * M_PI);
    cyc -= std::floor(cyc);
    p.carr_phase0 = static_cast<uint32_t>(
        static_cast<uint64_t>(cyc * 4294967296.0));
    p.carr_inc = carr.inc;

    p.chip_rate = eff_code_rate;
    p.g = sv.gain;
    p.L = sv.code_len;
    p.code = sv.code;
    p.nav = sv.nav;

    double eff = std::fmod(sv.code_phase0_chips, static_cast<double>(p.L));
    if (eff < 0) eff += p.L;
    eff = p.L - eff;
    if (eff >= p.L) eff -= p.L;
    p.eff = eff;
    p.cp_base = code_phase_at_seg + eff_code_rate * (p.abs_t0 - seg_t0);

    p.use_boc = sv.sub_carrier_hz > 0.0;
    p.cboc = sv.cboc;
    p.sec = sv.sec_code;
    p.sec_len = sv.sec_len;
    p.use_tx = sv.tx_time_valid != 0;
    p.tx_off = sv.tx_chips_offset;
    p.chips_per_sym = sv.code_rate_hz / nav_rate(p.nav);
    p.use_gk = sv.gain_nknots >= 2 && sv.gain_knots;
    p.inv_gdt = p.use_gk ? 1.0 / sv.gain_knot_dt : 0.0;
    p.gk_last = sv.gain_nknots - 2;
    p.cache_chip =
        p.chips_per_sym == std::floor(p.chips_per_sym) && p.chips_per_sym > 0.0;
    p.has_sec = p.sec_len > 0 && p.sec != nullptr;
    p.has_nav = p.nav.mode != NavMode::Zero && p.nav.nbits > 0;
    return p;
}

static inline void mix_sv_into(SvPrepared &p, double dt, int k0, int k1,
                               const std::complex<float> *__restrict lut,
                               float *__restrict iq) {
    const int L = p.L;
    const int8_t *__restrict code = p.code;
    const float g = p.g;
    const double abs_t0 = p.abs_t0;
    const double chip_rate = p.chip_rate;
    const double cp_base = p.cp_base;
    const double eff = p.eff;
    const bool use_traj = p.use_traj;
    uint32_t carr_phase = p.carr_phase0 + static_cast<uint32_t>(k0) * p.carr_inc;
    const uint32_t carr_inc = p.carr_inc;

    int64_t last_fl = p.last_fl;
    int64_t last_ufl = p.last_ufl;
    int64_t last_ci = p.last_ci;
    int64_t last_period = p.last_period;
    int64_t last_si = p.last_si;
    int64_t last_sym_idx = p.last_sym_idx;
    float last_nav_sym = p.last_nav_sym;
    float cached = p.cached;
    float cached_d = p.cached_d;

    // Fast path 1: no BOC, no gain knots, no secondary code, no nav bits
    if (!p.use_boc && !p.use_gk && !p.has_sec && !p.has_nav && p.cache_chip) {
        for (int k = k0; k < k1; ++k) {
            const double cp = use_traj ? (cp_base + chip_rate * (k * dt))
                                       : (eff + chip_rate * (abs_t0 + k * dt));
            const int64_t fl = static_cast<int64_t>(std::floor(cp));
            if (fl != last_fl) {
                int64_t ci;
                if (fl == last_fl + 1) {
                    ci = (last_ci + 1 == L) ? 0 : (last_ci + 1);
                } else {
                    ci = fl % L;
                    if (ci < 0) ci += L;
                }
                last_ci = ci;
                last_fl = fl;
                cached = static_cast<float>(code[ci]);
                cached_d = g * cached;
            }
            const std::complex<float> c = lut[(carr_phase + kLutRound) >> 20];
            carr_phase += carr_inc;
            iq[2 * k]     += cached_d * c.real();
            iq[2 * k + 1] += cached_d * c.imag();
        }
        p.last_fl = last_fl;
        p.last_ci = last_ci;
        p.cached = cached;
        p.cached_d = cached_d;
        return;
    }

    // Fast path 2: no BOC, no gain knots, tx_time_valid or secondary/nav active
    if (!p.use_boc && !p.use_gk && p.cache_chip && (p.use_tx || !p.has_nav)) {
        const bool use_tx = p.use_tx;
        const double tx_off = p.tx_off;
        const bool has_sec = p.has_sec;
        const int sec_len = p.sec_len;
        const int8_t *__restrict sec = p.sec;
        const bool has_tx_nav = use_tx && p.has_nav;
        const double chips_per_sym = p.chips_per_sym;
        const double dL = static_cast<double>(L);
        for (int k = k0; k < k1; ++k) {
            const double cp = use_traj ? (cp_base + chip_rate * (k * dt))
                                       : (eff + chip_rate * (abs_t0 + k * dt));
            const double u = use_tx ? (cp + tx_off) : cp;
            const int64_t fl = static_cast<int64_t>(std::floor(cp));
            const int64_t ufl = static_cast<int64_t>(std::floor(u));
            if (fl != last_fl || ufl != last_ufl) {
                if (fl != last_fl) {
                    int64_t ci;
                    if (fl == last_fl + 1) {
                        ci = (last_ci + 1 == L) ? 0 : (last_ci + 1);
                    } else {
                        ci = fl % L;
                        if (ci < 0) ci += L;
                    }
                    last_ci = ci;
                    last_fl = fl;
                }
                float chip = static_cast<float>(code[last_ci]);
                if (has_sec) {
                    const int64_t period = static_cast<int64_t>(std::floor(u / dL));
                    int64_t si;
                    if (period == last_period) {
                        si = last_si;
                    } else if (period == last_period + 1) {
                        si = (last_si + 1 == sec_len) ? 0 : (last_si + 1);
                        last_period = period;
                        last_si = si;
                    } else {
                        si = period % sec_len;
                        if (si < 0) si += sec_len;
                        last_period = period;
                        last_si = si;
                    }
                    chip *= static_cast<float>(sec[si]);
                }
                if (has_tx_nav) {
                    const int64_t sym_idx =
                        static_cast<int64_t>(std::floor(u / chips_per_sym));
                    if (sym_idx != last_sym_idx) {
                        last_nav_sym =
                            static_cast<float>(nav_symbol_at(p.nav, sym_idx));
                        last_sym_idx = sym_idx;
                    }
                    chip *= last_nav_sym;
                }
                cached = chip;
                cached_d = g * chip;
                last_ufl = ufl;
            }
            const std::complex<float> c = lut[(carr_phase + kLutRound) >> 20];
            carr_phase += carr_inc;
            iq[2 * k]     += cached_d * c.real();
            iq[2 * k + 1] += cached_d * c.imag();
        }
        p.last_fl = last_fl;
        p.last_ufl = last_ufl;
        p.last_ci = last_ci;
        p.last_period = last_period;
        p.last_si = last_si;
        p.last_sym_idx = last_sym_idx;
        p.last_nav_sym = last_nav_sym;
        p.cached = cached;
        p.cached_d = cached_d;
        return;
    }

    // General path (BOC / CBOC, fading gain knots, or legacy receiver-time nav)
    const SvChannel &sv = *p.sv;
    const bool use_tx = p.use_tx;
    const double tx_off = p.tx_off;
    const bool has_sec = p.has_sec;
    const int sec_len = p.sec_len;
    const int8_t *__restrict sec = p.sec;
    const double chips_per_sym = p.chips_per_sym;
    const bool cache_chip = p.cache_chip;
    const bool use_boc = p.use_boc;
    const int cboc = p.cboc;
    const bool use_gk = p.use_gk;
    const double inv_gdt = p.inv_gdt;
    const int64_t gk_last = p.gk_last;

    for (int k = k0; k < k1; ++k) {
        double t = abs_t0 + k * dt;
        double cp = use_traj ? (cp_base + chip_rate * (k * dt))
                             : (eff + chip_rate * t);
        const double u = use_tx ? (cp + tx_off) : cp;
        const int64_t fl = static_cast<int64_t>(std::floor(cp));
        const int64_t ufl = static_cast<int64_t>(std::floor(u));
        float chip;
        if (cache_chip && fl == last_fl && ufl == last_ufl) {
            chip = cached;
        } else {
            int64_t ci;
            if (fl == last_fl + 1) {
                ci = (last_ci + 1 == L) ? 0 : (last_ci + 1);
            } else {
                ci = fl % L;
                if (ci < 0) ci += L;
            }
            last_ci = ci;
            chip = static_cast<float>(code[ci]);
            if (has_sec) {
                int64_t period = static_cast<int64_t>(
                    std::floor(u / static_cast<double>(L)));
                int64_t si = period % sec_len;
                if (si < 0) si += sec_len;
                chip *= static_cast<float>(sec[si]);
            }
            if (use_tx && p.has_nav)
                chip *= static_cast<float>(nav_symbol_at(
                    p.nav, static_cast<int64_t>(std::floor(u / chips_per_sym))));
            cached = chip;
            last_fl = fl;
            last_ufl = ufl;
        }
        if (!use_tx && p.has_nav)
            chip *= static_cast<float>(nav_symbol(p.nav, t));
        if (use_boc) {
            int64_t hc = static_cast<int64_t>(std::floor(2.0 * cp));
            if (cboc != 0) {
                int64_t hc6 = static_cast<int64_t>(std::floor(12.0 * cp));
                const float s11 = (hc & 1) ? -1.0f : 1.0f;
                const float s61 = (hc6 & 1) ? -1.0f : 1.0f;
                chip *= kCbocA * s11 + static_cast<float>(cboc) * kCbocB * s61;
            } else if (hc & 1) {
                chip = -chip;
            }
        }
        std::complex<float> c = lut[(carr_phase + kLutRound) >> 20];
        carr_phase += carr_inc;
        if (use_gk) {
            const double x = t * inv_gdt - static_cast<double>(sv.gain_knot_j0);
            int64_t j = static_cast<int64_t>(std::floor(x));
            if (j < 0) j = 0;
            if (j > gk_last) j = gk_last;
            float f = static_cast<float>(x - static_cast<double>(j));
            f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
            const float *gp = sv.gain_knots + 2 * j;
            const float gr = gp[0] + f * (gp[2] - gp[0]);
            const float gi = gp[1] + f * (gp[3] - gp[1]);
            c = std::complex<float>(c.real() * gr - c.imag() * gi,
                                    c.real() * gi + c.imag() * gr);
        }
        float d = g * chip;
        iq[2 * k]     += d * c.real();
        iq[2 * k + 1] += d * c.imag();
    }
    p.last_fl = last_fl;
    p.last_ufl = last_ufl;
    p.last_ci = last_ci;
    p.cached = cached;
}

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__APPLE__) && !defined(_WIN32)
// target_clones needs ifunc (ELF/Mach-O symbol resolvers); MinGW's PE-COFF
// target does not support it -- the plain function below still compiles
// with whatever -march the build uses, just without runtime CPU dispatch.
__attribute__((target_clones("default", "sse4.2", "avx2", "avx2,fma")))
#endif
void mix_block(const SvChannel *__restrict svs, int nsv, double fs,
               uint64_t sample0, int n, float *__restrict iq) {
    if (nsv <= 0 || n <= 0) return;
    const double dt = 1.0 / fs;
    const std::complex<float> *__restrict lut = expLut().data();

    constexpr int kStackMax = 64;
    SvPrepared stack_prep[kStackMax];
    std::vector<SvPrepared> heap_prep;
    SvPrepared *prep = stack_prep;
    if (nsv > kStackMax) {
        heap_prep.resize(static_cast<size_t>(nsv));
        prep = heap_prep.data();
    }
    for (int s = 0; s < nsv; ++s) {
        prep[s] = prepare_sv(svs[s], fs, sample0);
    }

    // Tile over 2048-sample sub-blocks (16 KB of iq, fitting in L1D cache
    // across all nsv satellites) while preserving 100% IEEE-754 bit-identical
    // accumulation order and phase continuity.
    for (int k0 = 0; k0 < n; k0 += kMixTile) {
        const int k1 = (k0 + kMixTile < n) ? (k0 + kMixTile) : n;
        for (int s = 0; s < nsv; ++s) {
            mix_sv_into(prep[s], dt, k0, k1, lut, iq);
        }
    }
}

}  // namespace gs

#include "abi.h"
#include "scheduler.hpp"

static gs::SvChannel debug_sv(const int8_t *code, double code_rate,
                              double code_phase0, double code_doppler,
                              double carrier_freq, int sys = 0,
                              double sub_carrier_hz = 0.0,
                              const int8_t *sec_code = nullptr, int sec_len = 0,
                              double sec_rate_hz = 0.0) {
    gs::SvChannel sv;
    sv.code = code;
    sv.code_len = 1023;
    sv.code_rate_hz = code_rate;
    sv.code_phase0_chips = code_phase0;
    sv.code_doppler_hz = code_doppler;
    sv.carrier_freq_hz = carrier_freq;
    sv.gain = 1.0f;
    sv.sys = sys;
    sv.sub_carrier_hz = sub_carrier_hz;
    sv.sec_code = sec_code;
    sv.sec_len = sec_len;
    sv.sec_rate_hz = sec_rate_hz;
    return sv;
}

void synth_debug_one_sv(const int8_t *code, double code_rate, double code_phase0,
                        double code_doppler, double carrier_freq, double fs,
                        int n, float *iq) {
    for (int k = 0; k < 2 * n; ++k) iq[k] = 0.0f;
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq);
    gs::mix_block(&sv, 1, fs, 0, n, iq);
}

// Debug shim: one SV through gs::mix_block from an arbitrary absolute sample
// index, so tests can compare a single-shot run against a split-range run.
// Zeroes iq[0..2n) (mix_block is accumulate-only), then accumulates.
void synth_debug_mix_range(const int8_t *code, double code_rate,
                           double code_phase0, double code_doppler,
                           double carrier_freq, double fs, uint64_t sample0,
                           int n, float *iq) {
    for (int k = 0; k < 2 * n; ++k) iq[k] = 0.0f;
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq);
    gs::mix_block(&sv, 1, fs, sample0, n, iq);
}

// SP-B debug shim: one SV through gs::mix_block with the trajectory fields set.
void synth_debug_mix_traj(const int8_t *code, double code_rate,
                          double code_phase0, double carrier_freq, double fs,
                          uint64_t sample0, int n, int traj_nknots,
                          uint64_t traj_knot_samples,
                          const double *carr_freq, const double *carr_phase,
                          const double *code_rate_knots,
                          const double *code_phase_knots, float *iq) {
    for (int k = 0; k < 2 * n; ++k) iq[k] = 0.0f;
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, 0.0, carrier_freq);
    sv.traj_nknots = traj_nknots;
    sv.traj_knot_samples = traj_knot_samples;
    sv.traj_carr_freq = carr_freq;
    sv.traj_carr_phase = carr_phase;
    sv.traj_code_rate = code_rate_knots;
    sv.traj_code_phase = code_phase_knots;
    gs::mix_block(&sv, 1, fs, sample0, n, iq);
}

// ABI 28 debug shim: one SV through gs::mix_block_parallel with complex
// channel-gain knots, so tests can check the per-sample interpolation.
void synth_debug_mix_gain(const int8_t *code, double code_rate,
                          double code_phase0, double carrier_freq, double fs,
                          uint64_t sample0, int n, int nthreads, int nknots,
                          int64_t knot_j0, double knot_dt, const float *knots,
                          float *iq) {
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, 0.0, carrier_freq);
    sv.gain_nknots = nknots;
    sv.gain_knot_j0 = knot_j0;
    sv.gain_knot_dt = knot_dt;
    sv.gain_knots = knots;
    gs::mix_block_parallel(&sv, 1, fs, sample0, n, iq, nthreads);
}

// Debug shim: one SV through gs::mix_block_parallel (which zeroes iq itself),
// so tests can compare nthreads=1 against nthreads>1 over the same span.
void synth_debug_mix_parallel(const int8_t *code, double code_rate,
                              double code_phase0, double code_doppler,
                              double carrier_freq, double fs, uint64_t sample0,
                              int n, int nthreads, float *iq) {
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq);
    gs::mix_block_parallel(&sv, 1, fs, sample0, n, iq, nthreads);
}

// --- Task 10 _ex shims: expose sys / sub_carrier_hz / secondary code ---------

void synth_debug_one_sv_ex(const int8_t *code, double code_rate,
                           double code_phase0, double code_doppler,
                           double carrier_freq, double fs, int n,
                           int sys, double sub_carrier_hz,
                           const int8_t *sec_code, int sec_len,
                           double sec_rate_hz, float *iq) {
    for (int k = 0; k < 2 * n; ++k) iq[k] = 0.0f;
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq, sys, sub_carrier_hz, sec_code,
                                sec_len, sec_rate_hz);
    gs::mix_block(&sv, 1, fs, 0, n, iq);
}

void synth_debug_mix_range_ex(const int8_t *code, double code_rate,
                              double code_phase0, double code_doppler,
                              double carrier_freq, double fs, uint64_t sample0,
                              int n, int sys, double sub_carrier_hz,
                              const int8_t *sec_code, int sec_len,
                              double sec_rate_hz, float *iq) {
    for (int k = 0; k < 2 * n; ++k) iq[k] = 0.0f;
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq, sys, sub_carrier_hz, sec_code,
                                sec_len, sec_rate_hz);
    gs::mix_block(&sv, 1, fs, sample0, n, iq);
}

void synth_debug_mix_parallel_ex(const int8_t *code, double code_rate,
                                 double code_phase0, double code_doppler,
                                 double carrier_freq, double fs,
                                 uint64_t sample0, int n, int nthreads,
                                 int sys, double sub_carrier_hz,
                                 const int8_t *sec_code, int sec_len,
                                 double sec_rate_hz, float *iq) {
    gs::SvChannel sv = debug_sv(code, code_rate, code_phase0, code_doppler,
                                carrier_freq, sys, sub_carrier_hz, sec_code,
                                sec_len, sec_rate_hz);
    gs::mix_block_parallel(&sv, 1, fs, sample0, n, iq, nthreads);
}
