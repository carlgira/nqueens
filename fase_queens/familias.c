/*
 * familias.c — Familias de traslación: para cada solución S, qué desplazamientos de columna
 * t (mod N) dejan S+t como solución. Físicamente: todos los planetas a la misma velocidad,
 * el tablero gira entero; una racha de t consecutivos = fotos seguidas todas solución.
 *
 * Enumera todas las soluciones (backtracking con bits). Para cada una calcula:
 *   f   = número de t en [0,N) con S+t solución (familia),
 *   run = racha cíclica más larga de t consecutivos,
 *   col = colisiones de diagonales mod N (pares de filas en la misma diagonal "envuelta").
 * Se cuenta cada familia una vez por tablero (cada miembro la ve igual, con t desplazados).
 * Toroidales (f = N) aparte.
 *
 * uso: ./familias N   -> histogramas por pantalla, y ejemplos de la familia máxima
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int N; static int S[32];
static long hf[33], hr[33], hcol[64][33], tor, tot;
static int bestf, bestr; static int ex[4][32], exn, exT[4];

static int is_sol_shift(int t) {
    unsigned long long a = 0, b = 0;
    for (int r = 0; r < N; r++) {
        int c = S[r] + t; if (c >= N) c -= N;
        unsigned long long ma = 1ull << (r + c), mb = 1ull << (r - c + N);
        if ((a & ma) || (b & mb)) return 0; a |= ma; b |= mb;
    }
    return 1;
}
static void visit(void) {
    tot++;
    unsigned ok = 0; int f = 0;
    for (int t = 0; t < N; t++) if (t == 0 || is_sol_shift(t)) { ok |= 1u << t; f++; }
    if (f == N) { tor++; return; }
    int run = 0; for (int s = 0; s < N; s++) { int L = 0; while (L < N && (ok >> ((s + L) % N) & 1)) L++; if (L > run) run = L; }
    int col = 0; for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { if ((S[i] + i) % N == (S[j] + j) % N) col++; if ((S[i] - i + N * 2) % N == (S[j] - j + N * 2) % N) col++; }
    hf[f]++; hr[run]++; if (col < 64) hcol[col][f]++;
    if (run > bestr || (run == bestr && f > bestf)) { bestr = run; bestf = f; exn = 0; }
    if (run == bestr && f == bestf && exn < 4) { memcpy(ex[exn], S, sizeof(int) * N); exT[exn] = ok; exn++; }
}
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) { visit(); return; }
    unsigned free_ = ~cols & ((1u << N) - 1);
    while (free_) { int c = __builtin_ctz(free_); free_ &= free_ - 1;
        if (d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue;
        S[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N)); }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); bt(0, 0, 0, 0);
    printf("N=%d soluciones %ld, toroidales %ld\n", N, tot, tor);
    printf("  familia f (sin toroidales):"); for (int f = 1; f <= N; f++) if (hf[f]) printf(" %d:%ld", f, hf[f]); printf("\n");
    printf("  racha consecutiva:"); for (int f = 1; f <= N; f++) if (hr[f]) printf(" %d:%ld", f, hr[f]); printf("\n");
    printf("  familia media por colisiones mod N:");
    for (int c = 0; c < 64; c++) { long n = 0, s = 0; for (int f = 1; f <= N; f++) { n += hcol[c][f]; s += hcol[c][f] * f; } if (n) printf(" %d:%.2f(%ld)", c, (double)s / n, n); }
    printf("\n  maximo: racha %d, familia %d. Ejemplos:\n", bestr, bestf);
    for (int e = 0; e < exn; e++) { printf("    ["); for (int r = 0; r < N; r++) printf("%d%s", ex[e][r], r < N - 1 ? "," : ""); printf("]  t ="); for (int t = 0; t < N; t++) if (exT[e] >> t & 1) printf(" %d", t); printf("\n"); }
    return 0;
}
