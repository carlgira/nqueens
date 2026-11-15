/*
 * gcm.c — N-reinas como mapas acoplados (Kaneko) con acoplo repulsivo de ajedrez.
 *
 * Estado: x_r en [0,1), una por fila; posición continua u_r = N x_r; foto: c_r = floor(u_r).
 * Campo de ataque sobre la fila r en la columna c (núcleo gaussiano K de anchura sigma):
 *     A_r(c) = sum_{j!=r} K(c-u_j) + K((c+r)-(u_j+j)) + K((c-r)-(u_j-j))
 * Objetivo suave: y_r = sum_c c e^{-A_r(c)/tau} / sum_c e^{-A_r(c)/tau}
 * Paso, fila a fila en orden (cada fila ve ya movidas las anteriores; SINCRONO=1 para todas a la vez,
 * que oscila con periodo 2):
 *     x_r <- T_r + (1-eps)(f(x_r) - 1/2),  T_r = (y_r + 1/2)/N,  reflexión en los bordes
 * (1-eps) = amplitud del caos. KANEKO=1 usa la forma original (1-eps) f + eps T, cuyo término
 * caótico tiene media 1/2 y aparta a la reina del objetivo.
 * f: 0 = logístico a x(1-x) (Kaneko), 1 = tienda de pendiente a/2 (densidad uniforme si a=4).
 *
 * Mide: fracción de pasos cuya foto es solución (y en dos controles simétricos D4 de igual
 * tamaño, permutaciones al azar cerradas por D4), episodios (rachas de la misma foto),
 * permanencia, cobertura, exponente de Lyapunov máximo (Benettin), y para los saltos entre
 * soluciones consecutivas distintas: distancia de Hamming, misma clase D4, misma familia de
 * traslación (S' = S + t mod N), comparado con pares al azar de la misma lista de episodios.
 *
 * uso: ./gcm N eps [a=4] [tau=0.5] [sigma=0.5] [pasos=1e6] [semilla=1] [mapa=0] [lista_episodios.txt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MX 16
static int kaneko = 0; /* env KANEKO=1: forma original (1-eps)f + eps T */
static int N, mapa, seq = 1; /* env SINCRONO=1: todas las filas a la vez */ static double eps, A_, tau, sig, i2s2;
static uint64_t st = 88172645463325252ull;
static inline uint64_t rng(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
static inline double ur(void) { return (rng() >> 11) * (1.0 / 9007199254740992.0); }

/* ---- conjuntos de tableros (claves base N) ---- */
typedef struct { uint64_t *k; long n, cap; } Set;
static void sadd(Set *s, uint64_t k) { if (s->n == s->cap) { s->cap = s->cap ? 2 * s->cap : 1024; s->k = realloc(s->k, s->cap * 8); } s->k[s->n++] = k; }
static int cmpu(const void *a, const void *b) { uint64_t x = *(uint64_t *)a, y = *(uint64_t *)b; return x < y ? -1 : x > y; }
static void sfin(Set *s) { qsort(s->k, s->n, 8, cmpu); long m = 0; for (long i = 0; i < s->n; i++) if (!m || s->k[i] != s->k[m - 1]) s->k[m++] = s->k[i]; s->n = m; }
static long sfind(const Set *s, uint64_t k) { long lo = 0, hi = s->n - 1; while (lo <= hi) { long m = (lo + hi) / 2; if (s->k[m] == k) return m; if (s->k[m] < k) lo = m + 1; else hi = m - 1; } return -1; }
static uint64_t key(const int *c) { uint64_t k = 0; for (int r = N - 1; r >= 0; r--) k = k * N + c[r]; return k; }
static void unkey(uint64_t k, int *c) { for (int r = 0; r < N; r++) { c[r] = k % N; k /= N; } }
static void tr(const int *c, int t, int *o) { /* 8 simetrías D4 */
    for (int r = 0; r < N; r++) { int x = r, y = c[r], X, Y;
        switch (t) { case 0: X = x; Y = y; break; case 1: X = x; Y = N - 1 - y; break; case 2: X = N - 1 - x; Y = y; break; case 3: X = N - 1 - x; Y = N - 1 - y; break;
            case 4: X = y; Y = x; break; case 5: X = y; Y = N - 1 - x; break; case 6: X = N - 1 - y; Y = x; break; default: X = N - 1 - y; Y = N - 1 - x; }
        o[X] = Y; } }
static uint64_t canon(const int *c) { int o[MX]; uint64_t m = UINT64_MAX; for (int t = 0; t < 8; t++) { tr(c, t, o); uint64_t k = key(o); if (k < m) m = k; } return m; }
static Set SOL, CT[2]; static uint64_t *solcan;
static int tmp[MX];
static void bt(int r, unsigned cols, uint64_t d1, uint64_t d2) {
    if (r == N) { sadd(&SOL, key(tmp)); return; }
    unsigned f = ~cols & ((1u << N) - 1);
    while (f) { int c = __builtin_ctz(f); f &= f - 1; if (d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue; tmp[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N)); }
}
static int issol(const int *c) { unsigned a = 0; uint64_t b = 0, d = 0; for (int r = 0; r < N; r++) { if (a >> c[r] & 1 || b >> (r + c[r]) & 1 || d >> (r - c[r] + N) & 1) return 0; a |= 1u << c[r]; b |= 1ull << (r + c[r]); d |= 1ull << (r - c[r] + N); } return 1; }
static void make_ctrl(Set *s) { /* permutaciones al azar cerradas por D4, sin soluciones, tamaño >= |SOL| */
    int p[MX], o[MX];
    while (s->n < SOL.n) { for (int i = 0; i < N; i++) p[i] = i; for (int i = N - 1; i > 0; i--) { int j = rng() % (i + 1); int x = p[i]; p[i] = p[j]; p[j] = x; }
        if (issol(p)) continue; for (int t = 0; t < 8; t++) { tr(p, t, o); sadd(s, key(o)); } sfin(s); }
}

/* ---- dinámica ---- */
static inline double f(double x) { return mapa ? (x < 0.5 ? A_ * 0.5 * x : A_ * 0.5 * (1 - x)) : A_ * x * (1 - x); }
static void step(const double *x, double *xn) {
    double u[MX]; for (int r = 0; r < N; r++) u[r] = N * x[r];
    for (int r = 0; r < N; r++) {
        double Ar[MX], mn = 1e300;
        for (int c = 0; c < N; c++) { double s = 0;
            for (int j = 0; j < N; j++) { if (j == r) continue; double d0 = c + 0.5 - u[j], k = r - j;
                s += exp(-d0 * d0 * i2s2) + exp(-(d0 + k) * (d0 + k) * i2s2) + exp(-(d0 - k) * (d0 - k) * i2s2); }
            Ar[c] = s; if (s < mn) mn = s; }
        double W = 0, Y = 0; for (int c = 0; c < N; c++) { double w = exp(-(Ar[c] - mn) / tau); W += w; Y += w * c; }
        double T = (Y / W + 0.5) / N, v; /* candidatos en el centro de la casilla */
        if (kaneko) v = (1 - eps) * f(x[r]) + eps * T;   /* forma original: el caos arrastra hacia el centro */
        else { v = T + (1 - eps) * (f(x[r]) - 0.5);        /* agitación centrada en el objetivo, reflexión en bordes */
               if (v < 0) v = -v; if (v >= 1) v = 2 - v; }
        if (v < 0) v = 0; if (v >= 1) v = 1 - 1e-12; xn[r] = v; if (seq) u[r] = N * v; /* secuencial: las filas siguientes ya ven esta movida */
    }
}

typedef struct { long s; int len; } Ep;
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "uso: ./gcm N eps [a] [tau] [sigma] [pasos] [semilla] [mapa] [episodios.txt]\n"); return 1; }
    N = atoi(argv[1]); eps = atof(argv[2]); A_ = argc > 3 ? atof(argv[3]) : 4; tau = argc > 4 ? atof(argv[4]) : 0.5; sig = argc > 5 ? atof(argv[5]) : 0.5;
    long steps = argc > 6 ? (long)atof(argv[6]) : 1000000; long seed = argc > 7 ? atol(argv[7]) : 1; mapa = argc > 8 ? atoi(argv[8]) : 0;
    const char *epf = argc > 9 ? argv[9] : NULL;
    i2s2 = 1.0 / (2 * sig * sig);
    if (getenv("SINCRONO")) seq = 0;
    if (getenv("KANEKO")) kaneko = 1;
    bt(0, 0, 0, 0); sfin(&SOL);
    solcan = malloc(SOL.n * 8); for (long i = 0; i < SOL.n; i++) { int c[MX]; unkey(SOL.k[i], c); solcan[i] = canon(c); }
    st = 0x9E3779B97F4A7C15ull; make_ctrl(&CT[0]); st = 0xD1B54A32D192ED03ull; make_ctrl(&CT[1]);
    st = 0x9E3779B97F4A7C15ull * (uint64_t)seed + 7; for (int i = 0; i < 20; i++) rng();

    double x[MX], xn[MX], z[MX], zn[MX]; for (int r = 0; r < N; r++) x[r] = 0.05 + 0.9 * ur();
    for (int t = 0; t < 2000; t++) { step(x, xn); memcpy(x, xn, sizeof x); }
    const double d0 = 1e-9; double nv = 0; for (int r = 0; r < N; r++) { z[r] = ur() - 0.5; nv += z[r] * z[r]; } nv = sqrt(nv);
    for (int r = 0; r < N; r++) z[r] = x[r] + d0 * z[r] / nv;
    double lyap = 0;
    long onsol = 0, onct[2] = {0, 0}, *visit = calloc(SOL.n, sizeof(long));
    Ep *ep = malloc(sizeof(Ep) * 1024); long nep = 0, cap = 1024; long nepc[2] = {0, 0}; double lenc[2] = {0, 0};
    uint64_t prevk = UINT64_MAX; long prevs = -1; int prevc = -1, run = 0;
    for (long t = 0; t <= steps; t++) {
        int c[MX]; uint64_t k = UINT64_MAX; long s = -1; int cc = -1;
        if (t < steps) { for (int r = 0; r < N; r++) { c[r] = (int)(N * x[r]); if (c[r] >= N) c[r] = N - 1; } k = key(c); s = sfind(&SOL, k); if (s < 0) for (int q = 0; q < 2; q++) if (sfind(&CT[q], k) >= 0) cc = q; }
        if (k == prevk) run++;
        else { if (prevs >= 0) { if (nep == cap) { cap *= 2; ep = realloc(ep, sizeof(Ep) * cap); } ep[nep].s = prevs; ep[nep].len = run; nep++; }
               if (prevc >= 0) { nepc[prevc]++; lenc[prevc] += run; }
               prevk = k; prevs = s; prevc = cc; run = 1; }
        if (t == steps) break;
        if (s >= 0) { onsol++; visit[s]++; } if (cc >= 0) onct[cc]++;
        if (getenv("DEBUG") && t < atoi(getenv("DEBUG"))) { int at = 0; for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) at += c[i] == c[j] || c[i] - c[j] == i - j || c[i] - c[j] == j - i;
            printf("t=%ld [", t); for (int r = 0; r < N; r++) printf("%d", c[r]); printf("] ataques %d  u:", at); for (int r = 0; r < N; r++) printf(" %.2f", N * x[r]); printf("\n"); }
        /* paso + Lyapunov */
        step(x, xn); step(z, zn); double dd = 0; for (int r = 0; r < N; r++) { double e = zn[r] - xn[r]; dd += e * e; } dd = sqrt(dd);
        if (dd == 0) dd = 1e-300; lyap += log(dd / d0);
        for (int r = 0; r < N; r++) { x[r] = xn[r]; z[r] = xn[r] + (zn[r] - xn[r]) * d0 / dd; }
    }
    lyap /= steps;
    /* estadística */
    long dist = 0; double H = 0; for (long i = 0; i < SOL.n; i++) if (visit[i]) { dist++; double p = (double)visit[i] / onsol; H -= p * log(p); }
    double mlen = 0; for (long i = 0; i < nep; i++) mlen += ep[i].len; if (nep) mlen /= nep;
    /* saltos entre episodios consecutivos con solución distinta */
    long nt = 0; double ham = 0, scl = 0, str = 0;
    #define PAIR(A, B, h, cl, trn) { int ca[MX], cb[MX]; unkey(SOL.k[A], ca); unkey(SOL.k[B], cb); int hh = 0; for (int r = 0; r < N; r++) hh += ca[r] != cb[r]; h = hh; cl = solcan[A] == solcan[B]; \
        int tt = (cb[0] - ca[0] + N) % N, ok = 1; for (int r = 1; r < N; r++) if ((cb[r] - ca[r] + N) % N != tt) { ok = 0; break; } trn = ok; }
    for (long i = 0; i + 1 < nep; i++) if (ep[i].s != ep[i + 1].s) { int h, cl, trn; PAIR(ep[i].s, ep[i + 1].s, h, cl, trn); ham += h; scl += cl; str += trn; nt++; }
    double bham = 0, bcl = 0, btr = 0; long nb = 0;
    if (nep > 1) for (long q = 0; q < 200000; q++) { long i = rng() % nep, j = rng() % nep; if (ep[i].s == ep[j].s) continue; int h, cl, trn; PAIR(ep[i].s, ep[j].s, h, cl, trn); bham += h; bcl += cl; btr += trn; nb++; }
    printf("N=%d eps=%.3f a=%.2f tau=%.2f sig=%.2f mapa=%d | sol %.5f ctrl %.5f %.5f | episodios %ld (ctrl %ld %ld) perm %.2f (ctrl %.2f %.2f) | distintas %ld/%ld efectivas %.1f | lyap %.4f",
        N, eps, A_, tau, sig, mapa, (double)onsol / steps, (double)onct[0] / steps, (double)onct[1] / steps, nep, nepc[0], nepc[1], mlen,
        nepc[0] ? lenc[0] / nepc[0] : 0, nepc[1] ? lenc[1] / nepc[1] : 0, dist, SOL.n, exp(H), lyap);
    if (nt && nb) printf(" | saltos %ld: hamming %.2f (azar %.2f) misma_clase %.3f (%.3f) traslacion %.3f (%.3f)", nt, ham / nt, bham / nb, scl / nt, bcl / nb, str / nt, btr / nb);
    printf("\n");
    if (epf) { FILE *F = fopen(epf, "w"); for (long i = 0; i < nep; i++) { int c[MX]; unkey(SOL.k[ep[i].s], c); fprintf(F, "%d", ep[i].len); for (int r = 0; r < N; r++) fprintf(F, " %d", c[r]); fprintf(F, "\n"); } fclose(F); }
    return 0;
}
