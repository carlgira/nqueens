/*
 * ternas.c — La forma de las ternas de soluciones (enteras, mod N).
 *
 * Para cada terna ordenada (S0,S1,S2) de soluciones distintas y cada reina r:
 *     velocidad   V_r = S1_r - S0_r            (mod N)
 *     aceleración A_r = S2_r - 2 S1_r + S0_r   (mod N)
 * La "forma" de la terna es el vector A (N números).
 *
 * Hace tres cosas:
 *  1. Guarda TODAS las ternas en datos/ternas_N<N>_<conjunto>.bin para estudios posteriores
 *     (registro: i,j,k como uint16 + V[N] + A[N] como uint8) y la lista de tableros en .txt.
 *  2. Agrupa por vector A: cuántas formas distintas, las más repetidas, y por clases
 *     (cero, constante, afín en la fila, dos valores, casi cero).
 *  3. Crece cada terna con aceleración constante:  S_k = S0 + k V + k(k-1)/2 A  (mod N)
 *     mientras siga dando soluciones distintas. Cadena más larga.
 *
 * Conjuntos: 0 = soluciones; s>0 = control azar (Q permutaciones al azar, semilla s);
 *            s<0 = control simétrico (órbitas D4 de permutaciones al azar, semilla -s).
 *
 * uso: ./ternas N conjunto [guardar=1]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int N, Q, cap;
static int *sol;
static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) {
        if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); }
        memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return;
    }
    for (int c = 0; c < N; c++) {
        if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue;
        cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N));
    }
}
static unsigned long long rs = 12345;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static void randperm(int *p) { for (int r = 0; r < N; r++) p[r] = r; for (int r = N - 1; r > 0; r--) { int j = (int)(ur() * (r + 1)); int t = p[r]; p[r] = p[j]; p[j] = t; } }
static int have(const int *g, int n) { for (int k = 0; k < n; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) return 1; return 0; }
static void push(const int *g) { if (Q == cap) { cap = cap ? cap * 2 : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }

/* tabla hash: clave = vector A (codificado en base N, cabe en 64 bits para N<=16) */
#define HB 22
static uint64_t *hk; static uint32_t *hc;
static void hadd(uint64_t k) {
    uint64_t h = (k * 0x9E3779B97F4A7C15ULL) >> (64 - HB);
    while (hc[h] && hk[h] != k) h = (h + 1) & ((1u << HB) - 1);
    hk[h] = k; hc[h]++;
}
static int md(int x) { return ((x % N) + N) % N; }

/* clases del vector A */
static int cls(const int *A) {
    int nz = 0, allsame = 1, aff = 1, vals[2] = {-1, -1}, nv = 0;
    for (int r = 0; r < N; r++) {
        if (A[r]) nz++;
        if (A[r] != A[0]) allsame = 0;
        if (r >= 2 && md(A[r] - A[r - 1]) != md(A[1] - A[0])) aff = 0;
        if (A[r] != vals[0] && A[r] != vals[1]) { if (nv < 2) vals[nv] = A[r]; nv++; }
    }
    if (nz == 0) return 0;          /* aceleración cero: velocidad constante */
    if (allsame) return 1;          /* misma aceleración para todas */
    if (aff) return 2;              /* afín en la fila */
    if (nz <= 2) return 3;          /* casi cero: 1 o 2 reinas aceleran */
    if (nv <= 2) return 4;          /* solo dos valores */
    return 5;                       /* otros */
}
static const char *cname[6] = {"cero", "constante", "afin en fila", "1-2 defectos", "dos valores", "otros"};

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    int set = argc > 2 ? atoi(argv[2]) : 0;
    int save = argc > 3 ? atoi(argv[3]) : 1;
    bt(0, 0, 0, 0);
    int want = Q;
    if (set > 0) {
        rs ^= (unsigned long long)set * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32];
        while (Q < want) { randperm(p); if (!have(p, Q)) push(p); }
    } else if (set < 0) {
        rs ^= (unsigned long long)(-set) * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32], g[32], q[32];
        while (Q < want) {
            randperm(p);
            for (int e = 0; e < 8; e++) {
                memcpy(q, p, sizeof q);
                if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r;
                for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r];
                if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r];
                if (!have(g, Q)) push(g);
            }
        }
    }
    char tag[32]; if (set == 0) snprintf(tag, sizeof tag, "sol"); else if (set > 0) snprintf(tag, sizeof tag, "azar%d", set); else snprintf(tag, sizeof tag, "sim%d", -set);

    FILE *fb = NULL;
    if (save) {
        mkdir("datos", 0755);
        char fn[128];
        snprintf(fn, sizeof fn, "datos/tableros_N%d_%s.txt", N, tag);
        FILE *ft = fopen(fn, "w");
        for (int i = 0; i < Q; i++) { for (int r = 0; r < N; r++) fprintf(ft, "%d%c", sol[(long)i * N + r], r < N - 1 ? ' ' : '\n'); }
        fclose(ft);
        snprintf(fn, sizeof fn, "datos/ternas_N%d_%s.bin", N, tag);
        fb = fopen(fn, "wb");
        int32_t hdr[2] = {N, Q}; fwrite(hdr, sizeof hdr, 1, fb);
    }

    hk = calloc(1u << HB, sizeof(uint64_t)); hc = calloc(1u << HB, sizeof(uint32_t));
    long ncls[6] = {0}, total = 0;
    long chainhist[256] = {0}; int bestL = 0, bi = 0, bj = 0, bk = 0;
    long chaincls[6][64] = {{0}};
    int V[32], A[32], c[32];
    unsigned char *mark = calloc(Q, 1);
    /* índice para comprobar si un tablero está en el conjunto: búsqueda lineal es lenta; usamos hash de tableros */
    uint64_t *bk_ = malloc(sizeof(uint64_t) * Q);
    for (int i = 0; i < Q; i++) { uint64_t k = 0; for (int r = 0; r < N; r++) k = k * N + sol[(long)i * N + r]; bk_[i] = k; }
    /* tabla hash de tableros -> índice */
    int TB = 1; while (TB < 4 * Q) TB <<= 1;
    int64_t *tk = malloc(sizeof(int64_t) * TB); int *ti = malloc(sizeof(int) * TB);
    for (int h = 0; h < TB; h++) ti[h] = -1;
    for (int i = 0; i < Q; i++) { uint64_t h = (bk_[i] * 0x9E3779B97F4A7C15ULL) & (TB - 1); while (ti[h] >= 0) h = (h + 1) & (TB - 1); tk[h] = bk_[i]; ti[h] = i; }

    for (int i = 0; i < Q; i++) for (int j = 0; j < Q; j++) {
        if (j == i) continue;
        const int *S0 = sol + (long)i * N, *S1 = sol + (long)j * N;
        for (int r = 0; r < N; r++) V[r] = md(S1[r] - S0[r]);
        for (int k = 0; k < Q; k++) {
            if (k == i || k == j) continue;
            const int *S2 = sol + (long)k * N;
            uint64_t key = 0;
            for (int r = 0; r < N; r++) { A[r] = md(S2[r] - 2 * S1[r] + S0[r]); key = key * N + A[r]; }
            hadd(key); total++;
            int cl = cls(A); ncls[cl]++;
            if (fb) {
                uint16_t ijk[3] = {(uint16_t)i, (uint16_t)j, (uint16_t)k};
                unsigned char vv[32], aa[32];
                for (int r = 0; r < N; r++) { vv[r] = (unsigned char)V[r]; aa[r] = (unsigned char)A[r]; }
                fwrite(ijk, sizeof ijk, 1, fb); fwrite(vv, 1, N, fb); fwrite(aa, 1, N, fb);
            }
            /* crecer con aceleración constante */
            int used[4096]; int nu = 3; used[0] = i; used[1] = j; used[2] = k;
            int L = 3;
            for (int s = 3; s < 4096; s++) {
                uint64_t kk = 0;
                for (int r = 0; r < N; r++) { c[r] = md(S0[r] + s * V[r] + (s * (s - 1) / 2) % N * A[r]); kk = kk * N + c[r]; }
                uint64_t h = (kk * 0x9E3779B97F4A7C15ULL) & (TB - 1); int idx = -1;
                while (ti[h] >= 0) { if ((uint64_t)tk[h] == kk) { idx = ti[h]; break; } h = (h + 1) & (TB - 1); }
                if (idx < 0) break;
                int dup = 0; for (int u = 0; u < nu; u++) if (used[u] == idx) dup = 1;
                if (dup) break;
                used[nu++] = idx; L++;
            }
            if (L < 256) chainhist[L]++;
            if (L < 64) chaincls[cl][L]++;
            if (L > bestL) { bestL = L; bi = i; bj = j; bk = k; }
        }
    }
    if (fb) fclose(fb);

    /* estadísticas de formas */
    long distinct = 0; uint32_t mx = 0; double h2 = 0;
    for (uint32_t h = 0; h < (1u << HB); h++) if (hc[h]) { distinct++; if (hc[h] > mx) mx = hc[h]; h2 += (double)hc[h] * hc[h]; }
    printf("N=%d conjunto=%s Q=%d ternas=%ld\n", N, tag, Q, total);
    printf("  formas (vectores A) distintas: %ld   la mas repetida: %u ternas   repeticion media: %.2f\n", distinct, mx, h2 / total);
    printf("  por clase:");
    for (int k = 0; k < 6; k++) printf("  %s %ld", cname[k], ncls[k]);
    printf("\n  crecimiento con aceleracion constante, cadenas por longitud:");
    for (int L = 3; L < 256; L++) if (chainhist[L]) printf("  %d:%ld", L, chainhist[L]);
    printf("\n  longitud maxima %d  (terna %d,%d,%d)\n", bestL, bi, bj, bk);
    printf("  longitud >=5 por clase de A:");
    for (int k = 0; k < 6; k++) { long s = 0; for (int L = 5; L < 64; L++) s += chaincls[k][L]; printf("  %s %ld", cname[k], s); }
    printf("\n");
    /* top 8 formas */
    printf("  formas mas repetidas:\n");
    for (int t = 0; t < 8; t++) {
        uint32_t bm = 0, bh = 0;
        for (uint32_t h = 0; h < (1u << HB); h++) if (hc[h] > bm) { bm = hc[h]; bh = h; }
        if (!bm) break;
        uint64_t k = hk[bh]; int a[32];
        for (int r = N - 1; r >= 0; r--) { a[r] = k % N; k /= N; }
        printf("    %5u ternas  A=", bm);
        for (int r = 0; r < N; r++) printf("%d", a[r]);
        printf("  (%s)\n", cname[cls(a)]);
        hc[bh] = 0;
    }
    return 0;
}
