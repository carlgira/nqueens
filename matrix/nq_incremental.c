// Tabla de atacantes: reconstruirla en cada solución frente a actualizarla con el movimiento.
// Graba un paseo aleatorio por vecinos Cayley<=2 y lo reproduce con cada método, comparando coste y resultado.
// Estado que necesita el generador: att[r][c] y, por jugada a->b, cuántas filas extra exige (0, 1 o más).
// Tercer método: sin tabla, solo ocupación de diagonales actualizada y las N(N-1) jugadas reclasificadas.
// Uso: ./nq_incremental N pasos semilla
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static int N;
static uint8_t *S; static size_t CNT;
static void gen(int r, uint32_t cols, uint32_t d1, uint32_t d2, uint8_t *q) {
    uint32_t full = (1u << N) - 1;
    if (r == N) { memcpy(S + CNT * 16, q, N); CNT++; return; }
    uint32_t av = ~(cols | d1 | d2) & full;
    while (av) {
        uint32_t bit = av & -av; av ^= bit; q[r] = (uint8_t)__builtin_ctz(bit);
        gen(r + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1, q);
    }
}
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

typedef struct { uint32_t att[16][16], live0[16], live1[16]; uint8_t q[16], own[16]; } St;
static long long ops;

static inline void classify(St *s, int a, int b) {      // jugada a->b: la fila a toma la columna de b
    uint32_t e = s->att[a][s->q[b]] & ~(1u << b); int pc = __builtin_popcount(e); uint32_t m = 1u << b;
    s->live0[a] = pc == 0 ? s->live0[a] | m : s->live0[a] & ~m;
    s->live1[a] = pc <= 1 ? s->live1[a] | m : s->live1[a] & ~m; ops++;
}
static void build(St *s, const uint8_t *q) {
    int o1[32], o2[32]; memset(o1, -1, sizeof o1); memset(o2, -1, sizeof o2);
    memcpy(s->q, q, 16);
    for (int r = 0; r < N; r++) { s->own[q[r]] = r; o1[r - q[r] + N - 1] = r; o2[r + q[r]] = r; s->live0[r] = s->live1[r] = 0; }
    for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) {
        uint32_t m = 0; int o = o1[r - c + N - 1]; if (o >= 0 && o != r) m |= 1u << o;
        o = o2[r + c]; if (o >= 0 && o != r) m |= 1u << o; s->att[r][c] = m; ops++;
    }
    for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) if (a != b) classify(s, a, b);
}
static uint32_t dirty[16];                               // celdas tocadas, una máscara de columnas por fila
static inline void flip(St *s, int r0, int c0) {         // pone o quita la reina (r0,c0) en sus dos diagonales
    uint32_t b = 1u << r0;
    for (int r = 0; r < N; r++) if (r != r0) {
        int d = r - r0, c = c0 + d;
        if (c >= 0 && c < N) { s->att[r][c] ^= b; dirty[r] |= 1u << c; ops++; }
        c = c0 - d;
        if (c >= 0 && c < N) { s->att[r][c] ^= b; dirty[r] |= 1u << c; ops++; }
    }
}
static void update(St *s, const uint8_t *q2, const int *rows, int k) {
    memset(dirty, 0, sizeof dirty);
    for (int t = 0; t < k; t++) flip(s, rows[t], s->q[rows[t]]);
    for (int t = 0; t < k; t++) flip(s, rows[t], q2[rows[t]]);
    uint32_t cols = 0;
    for (int t = 0; t < k; t++) { int r = rows[t]; s->q[r] = q2[r]; s->own[q2[r]] = r; cols |= 1u << q2[r]; }
    for (int r = 0; r < N; r++) dirty[r] |= cols;        // las columnas movidas cambian de dueño
    for (int r = 0; r < N; r++) { uint32_t v = dirty[r]; while (v) { int c = __builtin_ctz(v); v &= v - 1; if (s->own[c] != r) classify(s, r, s->own[c]); } }
}
static int same(const St *a, const St *b) {
    return !memcmp(a->att, b->att, sizeof a->att) && !memcmp(a->live0, b->live0, sizeof a->live0) && !memcmp(a->live1, b->live1, sizeof a->live1);
}
static int valid(const uint8_t *q) {
    uint32_t a = 0, b = 0;
    for (int i = 0; i < N; i++) { uint32_t x = 1u << (i - q[i] + N - 1), y = 1u << (i + q[i]); if ((a & x) || (b & y)) return 0; a |= x; b |= y; }
    return 1;
}
typedef struct { uint8_t k, rows[4], cols[4]; } Mv;
static int neighbours(const uint8_t *q, Mv *out) {
    int n = 0; uint8_t t[16];
#define TRY(K, ...) do { int r_[] = {__VA_ARGS__}; memcpy(t, q, 16); for (int x = 0; x < K; x++) t[r_[x]] = q[r_[K + x]]; \
        if (valid(t)) { out[n].k = K; for (int x = 0; x < K; x++) { out[n].rows[x] = r_[x]; out[n].cols[x] = t[r_[x]]; } n++; } } while (0)
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { TRY(2, i, j, j, i);
        for (int k = j + 1; k < N; k++) { TRY(3, i, j, k, j, k, i); TRY(3, i, j, k, k, i, j);
            for (int l = k + 1; l < N; l++) { TRY(4, i, j, k, l, j, i, l, k); TRY(4, i, j, k, l, k, l, i, j); TRY(4, i, j, k, l, l, k, j, i); } } }
    return n;
}
typedef struct { int8_t o1[32], o2[32]; uint8_t q[16]; uint32_t live0[16], live1[16]; } Oc;
static void oc_init(Oc *s, const uint8_t *q) {
    memset(s->o1, -1, sizeof s->o1); memset(s->o2, -1, sizeof s->o2); memcpy(s->q, q, 16);
    for (int r = 0; r < N; r++) { s->o1[r - q[r] + N - 1] = r; s->o2[r + q[r]] = r; }
}
static void oc_move(Oc *s, const uint8_t *q2, const int *rows, int k) {
    for (int t = 0; t < k; t++) { int r = rows[t], c = s->q[r]; s->o1[r - c + N - 1] = -1; s->o2[r + c] = -1; ops += 2; }
    for (int t = 0; t < k; t++) { int r = rows[t], c = q2[r]; s->q[r] = c; s->o1[r - c + N - 1] = r; s->o2[r + c] = r; ops += 2; }
}
static void oc_classify(Oc *s) {
    for (int a = 0; a < N; a++) { uint32_t l0 = 0, l1 = 0;
        for (int b = 0; b < N; b++) if (a != b) { int c = s->q[b], x = s->o1[a - c + N - 1], y = s->o2[a + c];
            int pc = (x >= 0 && x != a && x != b) + (y >= 0 && y != a && y != b);
            l0 |= (uint32_t)(pc == 0) << b; l1 |= (uint32_t)(pc <= 1) << b; ops++; }
        s->live0[a] = l0; s->live1[a] = l1; }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    N = atoi(argv[1]); int M = atoi(argv[2]); rs ^= (uint64_t)atoi(argv[3]) * 0x9E3779B97F4A7C15ULL;
    S = malloc(15000000ULL * 16); uint8_t q0[16] = {0}; gen(0, 0, 0, 0, q0);
    uint8_t (*W)[16] = malloc((M + 1) * 16); Mv *mv = malloc((M + 1) * sizeof(Mv)); Mv nb[512];
    uint8_t cur[16]; memcpy(cur, S + rnd() % CNT * 16, 16); memcpy(W[0], cur, 16);
    int restarts = 0; double ks = 0;
    for (int i = 0; i < M; i++) {                        // k = 0 marca un salto a otra solución (sin vecinos o cada 1.000 pasos)
        int n = (i + 1) % 1000 ? neighbours(cur, nb) : 0;
        if (!n) { memcpy(cur, S + rnd() % CNT * 16, 16); mv[i].k = 0; restarts++; }
        else { Mv m = nb[rnd() % n]; mv[i] = m; ks += m.k; for (int x = 0; x < m.k; x++) cur[m.rows[x]] = m.cols[x]; }
        memcpy(W[i + 1], cur, 16);
    }
    static St a, b;
    ops = 0; double t0 = now();
    for (int rep = 0; rep < 20; rep++) for (int i = 0; i <= M; i++) build(&a, W[i]);
    double tf = (now() - t0) / 20 / (M + 1), of = (double)ops / 20 / (M + 1);
    ops = 0; t0 = now(); long long steps = 0;
    for (int rep = 0; rep < 20; rep++) { long long sv = ops; build(&b, W[0]); ops = sv;
        for (int i = 0; i < M; i++) {
            if (!mv[i].k) { sv = ops; build(&b, W[i + 1]); ops = sv; continue; }
            int r[4]; for (int x = 0; x < mv[i].k; x++) r[x] = mv[i].rows[x];
            update(&b, W[i + 1], r, mv[i].k); steps++;
        } }
    double ti = (now() - t0) / steps, oi = (double)ops / steps;
    static Oc o; long long sink = 0;
    ops = 0; t0 = now();
    for (int rep = 0; rep < 20; rep++) { long long sv = ops; oc_init(&o, W[0]); ops = sv;
        for (int i = 0; i < M; i++) {
            if (!mv[i].k) { oc_init(&o, W[i + 1]); continue; }
            int r[4]; for (int x = 0; x < mv[i].k; x++) r[x] = mv[i].rows[x];
            oc_move(&o, W[i + 1], r, mv[i].k); oc_classify(&o); sink += o.live1[0];
        } }
    double tc = (now() - t0) / steps, oc = (double)ops / steps;
    long long bad = 0; build(&b, W[0]);
    for (int i = 0; i < M; i++) { if (!mv[i].k) { build(&b, W[i + 1]); continue; }
        int r[4]; for (int x = 0; x < mv[i].k; x++) r[x] = mv[i].rows[x];
        update(&b, W[i + 1], r, mv[i].k); build(&a, W[i + 1]); bad += !same(&a, &b);
        oc_init(&o, W[i + 1]); oc_classify(&o); bad += memcmp(o.live0, a.live0, sizeof o.live0) || memcmp(o.live1, a.live1, sizeof o.live1); }
    printf("N=%d pasos=%d saltos=%d filas movidas=%.2f | reconstruir: %.0f ops, %.0f ns | incremental: %.0f ops, %.0f ns | sin tabla: %.0f ops, %.0f ns | diferencias=%lld (%lld)\n",
           N, M, restarts, ks / (M - restarts), of, tf * 1e9, oi, ti * 1e9, oc, tc * 1e9, bad, sink & 1);
}
