/*
 * rachas.c — Búsqueda directa de soluciones S tales que S, S+1, ..., S+(R-1) (columnas mod N)
 * son TODAS soluciones: R fotos seguidas buenas cuando todos los planetas van a la misma
 * velocidad. En vez de enumerar todas las soluciones (imposible para N>=18), se colocan las
 * reinas fila a fila comprobando a la vez las R copias desplazadas (poda muy fuerte).
 *
 * Cuenta cada racha por su tablero inicial (t=0). Una familia con racha exacta L >= R
 * aparece L-R+1 veces. Con limite de tiempo, el conteo es una cota inferior.
 *
 * uso: ./rachas N R [limite_s=60] [max_ejemplos=3] [fijas "r:c,r:c" (r<0 cuenta desde abajo)]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int N, R, S[64], maxex; static long cnt, nodes, ntor; static double lim; static clock_t t0; static int to;
static int fix[64]; /* -1 libre, si no columna fija */
static unsigned long long d1[64], d2[64]; /* máscara de diagonales por copia t */

static void dfs(int r, unsigned cols) {
    if (to) return;
    if ((++nodes & 0xfffff) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > lim) { to = 1; return; }
    if (r == N) {
        unsigned long long a = 0, b = 0; for (int i = 0; i < N; i++) { a |= 1ull << ((S[i] + i) % N); b |= 1ull << ((S[i] - i + N) % N); }
        if (a == (1ull << N) - 1 && b == (1ull << N) - 1) { ntor++; return; } /* toroidal: aparte */
        if (cnt < maxex) { printf("  ["); for (int i = 0; i < N; i++) printf("%d%s", S[i], i < N - 1 ? "," : ""); printf("]\n"); }
        cnt++; return;
    }
    unsigned fr = ~cols & ((1u << N) - 1);
    if (fix[r] >= 0) fr &= 1u << fix[r];
    while (fr) {
        int c = __builtin_ctz(fr); fr &= fr - 1; int ok = 1;
        for (int t = 0; t < R; t++) { int ct = c + t; if (ct >= N) ct -= N; if (d1[t] >> (r + ct) & 1 || d2[t] >> (r - ct + N) & 1) { ok = 0; break; } }
        if (!ok) continue;
        for (int t = 0; t < R; t++) { int ct = c + t; if (ct >= N) ct -= N; d1[t] ^= 1ull << (r + ct); d2[t] ^= 1ull << (r - ct + N); }
        S[r] = c; dfs(r + 1, cols | 1u << c);
        for (int t = 0; t < R; t++) { int ct = c + t; if (ct >= N) ct -= N; d1[t] ^= 1ull << (r + ct); d2[t] ^= 1ull << (r - ct + N); }
        if (to) return;
    }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); R = atoi(argv[2]); lim = argc > 3 ? atof(argv[3]) : 60; maxex = argc > 4 ? atoi(argv[4]) : 3;
    for (int r = 0; r < N; r++) fix[r] = -1;
    if (argc > 5) { char *p = argv[5]; while (*p) { int r, c, n; if (sscanf(p, "%d:%d%n", &r, &c, &n) != 2) break; if (r < 0) r += N; fix[r] = c; p += n; if (*p == ',') p++; } }
    t0 = clock(); dfs(0, 0);
    printf("N=%d R=%d: %ld tableros iniciales (no toroidales; toroidales aparte %ld)%s  (%.1fs)\n", N, R, cnt, ntor, to ? " [tiempo agotado: cota inferior]" : "", (double)(clock() - t0) / CLOCKS_PER_SEC);
    return 0;
}
