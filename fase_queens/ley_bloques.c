/*
 * ley_bloques.c — Búsqueda guiada por la ley encontrada en N=16.
 *
 * Ley: todas las reinas con la MISMA velocidad real v, y algunas además saltan N/4 por paso:
 *        c_r(k) = floor(beta_r + k v) + k (N/4) p_r      (mod N),   p_r en {0,1}
 * Patrón p: igual en las dos mitades (p[r+N/2] = p[r]); con espejo=1 además simétrico
 * dentro de cada bloque (p[r] = p[N/2-1-r]).
 *
 * Para cada patrón p y cada tablero de partida: DFS de cadenas de tableros distintos del
 * conjunto que siguen la ley (factibilidad exacta de v común: intervalo de recta digital por
 * reina e intersección). Guarda por patrón la cadena más larga y el grupo que forma.
 *
 * uso: ./ley_bloques N tableros.txt [espejo=1] [limite_s=600]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXK 128
static int N, Q, H, Q4;
static int *sol;
static int p[64];
typedef struct { double L, U; long x[MAXK]; int ok; } Tr;
static Tr tr[MAXK][64][2];
static int chain[MAXK], K, best, bestchain[MAXK];
static char *used;
static uint64_t *keys; static int TB; static int *ti;
static double tlimit; static clock_t t0; static int timed_out; static long nodes;

static uint64_t enc(const int *c) { uint64_t k = 1469598103934665603ULL; for (int r = 0; r < N; r++) { k ^= (uint64_t)c[r] + 1; k *= 1099511628211ULL; } return k; }

static int ext1(const Tr *a, Tr *b, int k, long xk) {
    if (!a->ok) { b->ok = 0; return 0; }
    *b = *a; b->x[k] = xk;
    for (int i = 0; i < k; i++) { double d = k - i, dx = (double)(xk - b->x[i]); double lo = (dx - 1) / d, hi = (dx + 1) / d; if (lo > b->L) b->L = lo; if (hi < b->U) b->U = hi; }
    b->ok = b->L < b->U - 1e-12; return b->ok;
}
/* ¿intervalo común de v entre reinas? cada reina: unión de 2 intervalos */
static int common(int k) {
    for (int r0 = 0; r0 < N; r0++) for (int s0 = 0; s0 < 2; s0++) {
        const Tr *t0 = &tr[k][r0][s0]; if (!t0->ok) continue;
        double a = t0->L + 1e-10; int good = 1;
        for (int r = 0; r < N && good; r++) { int any = 0; for (int s = 0; s < 2; s++) { const Tr *t = &tr[k][r][s]; if (t->ok && a > t->L && a < t->U) any = 1; } if (!any) good = 0; }
        if (good) return 1;
    }
    return 0;
}
/* intenta añadir el tablero T en la posición k */
static int try_add(int k, const int *P, const int *T) {
    for (int r = 0; r < N; r++) {
        int d = ((T[r] - P[r] - Q4 * p[r]) % N + 2 * N) % N; int any = 0;
        for (int s = 0; s < 2; s++) { long inc = (s == 1 && d == 0) ? N : d; if (ext1(&tr[k - 1][r][s], &tr[k][r][s], k, tr[k - 1][r][s].x[k - 1] + inc)) any = 1; }
        if (!any) return 0;
    }
    return common(k);
}
static void dfs(void) {
    if (timed_out) return;
    if ((++nodes & 0xff) == 0 && (double)(clock() - t0) / CLOCKS_PER_SEC > tlimit) { timed_out = 1; return; }
    int ext = 0;
    if (K < MAXK) {
        const int *P = sol + (long)chain[K - 1] * N;
        for (int j = 0; j < Q; j++) {
            if (used[j]) continue;
            if (!try_add(K, P, sol + (long)j * N)) continue;
            ext = 1; chain[K++] = j; used[j] = 1; dfs(); used[j] = 0; K--;
            if (timed_out) return;
        }
    }
    if (!ext && K > best) { best = K; memcpy(bestchain, chain, sizeof(int) * K); }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); H = N / 2; Q4 = N / 4;
    int espejo = argc > 3 ? atoi(argv[3]) : 1; tlimit = argc > 4 ? atof(argv[4]) : 600;
    FILE *f = fopen(argv[2], "r"); int cap = 1024; sol = malloc(sizeof(int) * N * cap); Q = 0;
    while (1) { if (Q == cap) { cap *= 2; sol = realloc(sol, sizeof(int) * N * cap); } int ok = 1; for (int r = 0; r < N; r++) if (fscanf(f, "%d", &sol[(long)Q * N + r]) != 1) { ok = 0; break; } if (!ok) break; Q++; }
    fclose(f); used = calloc(Q, 1);
    int free_bits = espejo ? H / 2 : H;
    printf("N=%d Q=%d patrones %s: %d\n", N, Q, espejo ? "con espejo" : "sin espejo", 1 << free_bits);
    t0 = clock();
    int gbest = 0;
    for (int m = 0; m < (1 << free_bits) && !timed_out; m++) {
        for (int r = 0; r < H; r++) { int b = espejo ? (r < H / 2 ? r : H - 1 - r) : r; p[r] = (m >> b) & 1; p[r + H] = p[r]; }
        /* p y su complemento son equivalentes (v <-> v + N/4): nos quedamos con p[0]=0 */
        if (p[0] == 1) continue;
        best = 0;
        for (int i = 0; i < Q && !timed_out; i++) {
            for (int r = 0; r < N; r++) for (int s = 0; s < 2; s++) { Tr *t = &tr[0][r][s]; t->L = 0; t->U = N; t->ok = 1; t->x[0] = sol[(long)i * N + r]; }
            /* v en [0, N/4): el resto lo aportan los saltos N/4 (y el complemento del patrón) */
            chain[0] = i; used[i] = 1; K = 1; dfs(); used[i] = 0;
        }
        if (best >= 4) {
            printf("  patron ");
            for (int r = 0; r < H; r++) printf("%d", p[r]);
            printf(": cadena maxima %d", best);
            /* v de la mejor cadena */
            printf("  inicio [");
            for (int r = 0; r < N; r++) printf("%d%s", sol[(long)bestchain[0] * N + r], r < N - 1 ? "," : "");
            printf("]\n"); fflush(stdout);
        }
        if (best > gbest) gbest = best;
    }
    printf("maximo global %d%s  (%.0fs, nodos %ld)\n", gbest, timed_out ? " [tiempo agotado]" : "", (double)(clock() - t0) / CLOCKS_PER_SEC, nodes);
    return 0;
}
