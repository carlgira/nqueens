/*
 * cadenas_frac.c — Patrón 1 con velocidades NO enteras.
 *
 * Cada reina r es una onda leída por su fase, con velocidad real alpha_r (columnas por paso):
 *        c_r(k) = floor(alpha_r * k + beta_r)   (mod N)
 * Buscamos cadenas de soluciones distintas S_0, S_1, ..., S_{K-1} (N-reinas estándar) tales
 * que, para cada reina, existe (alpha_r, beta_r) que reproduce su secuencia de columnas.
 *
 * Condición exacta: tomando alpha en [0, N), la posición sin envolver x_k crece en incrementos
 * delta_k = c(k+1)-c(k) mod N (un 0 puede ser también N si alpha ~ N). Existe beta sii
 *        L = max_{i<j} (x_j-x_i-1)/(j-i)  <  U = min_{i<j} (x_j-x_i+1)/(j-i),
 * y entonces cualquier alpha en (L,U) sirve. Es la condición de "segmento de recta digital".
 *
 * Dos modos:  libre = cada reina con su propia alpha_r
 *             comun = una sola alpha para todas (fases distintas): la intersección no vacía.
 *
 * Búsqueda: DFS exhaustiva desde cada par (S_0, S_1).
 * uso: ./cadenas_frac N modo(0 libre,1 comun) [max_ejemplos]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXK 64
static int N, Q, cap, MODE;
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

/* estado por reina y por interpretación (0: incrementos en [0,N-1]; 1: los ceros cuentan como N) */
typedef struct { double L, U; long x[MAXK]; int ok; } Track;
static Track tr[MAXK][32][2]; /* [profundidad][reina][interp] */
static int chain[MAXK], K;
static long histmax[MAXK];
static int best = 0, nbest = 0, shown = 0, maxshow;
static char *used;

/* extiende el track con nuevo punto x_K; devuelve ok */
static int extend(const Track *a, Track *b, int k, long xk) {
    if (!a->ok) { b->ok = 0; return 0; }
    *b = *a; b->x[k] = xk;
    for (int i = 0; i < k; i++) {
        double d = (double)(k - i), dx = (double)(xk - b->x[i]);
        double lo = (dx - 1) / d, hi = (dx + 1) / d;
        if (lo > b->L) b->L = lo;
        if (hi < b->U) b->U = hi;
    }
    b->ok = b->L < b->U - 1e-12;
    return b->ok;
}

/* ¿hay alpha común a todas las reinas? cada reina aporta la unión de hasta 2 intervalos */
static int common_alpha(int k, double *aout) {
    /* candidatos: extremos L de todos los intervalos (+eps); comprobamos cada uno */
    for (int r0 = 0; r0 < N; r0++) for (int s0 = 0; s0 < 2; s0++) {
        const Track *t0 = &tr[k][r0][s0]; if (!t0->ok) continue;
        /* probamos el punto medio de [max L, min U] restringido a este intervalo: barrido simple */
        double a = t0->L + 1e-9;
        int good = 1;
        for (int r = 0; r < N && good; r++) {
            int any = 0;
            for (int s = 0; s < 2; s++) { const Track *t = &tr[k][r][s]; if (t->ok && a > t->L && a < t->U) any = 1; }
            if (!any) good = 0;
        }
        if (good) { if (aout) *aout = a; return 1; }
    }
    return 0;
}

static void report(void) {
    printf("   K=%d:", K);
    for (int k = 0; k < K; k++) {
        printf(" [");
        for (int r = 0; r < N; r++) printf("%d%s", sol[(long)chain[k] * N + r], r < N - 1 ? "," : "");
        printf("]");
    }
    printf("\n     alpha por reina:");
    for (int r = 0; r < N; r++) {
        int s = tr[K - 1][r][0].ok ? 0 : 1;
        printf(" (%.3f,%.3f)", tr[K - 1][r][s].L, tr[K - 1][r][s].U);
    }
    if (MODE) { double a; if (common_alpha(K - 1, &a)) printf("  comun~%.4f", a); }
    printf("\n");
}

static void dfs(void) {
    int k = K; /* próximo índice */
    int ext = 0;
    if (k < MAXK) {
        const int *P = sol + (long)chain[k - 1] * N;
        for (int j = 0; j < Q; j++) {
            if (used[j]) continue;
            const int *T = sol + (long)j * N;
            int ok = 1;
            for (int r = 0; r < N && ok; r++) {
                int d = ((T[r] - P[r]) % N + N) % N;
                int any = 0;
                for (int s = 0; s < 2; s++) {
                    long inc = (s == 1 && d == 0) ? N : d;
                    if (extend(&tr[k - 1][r][s], &tr[k][r][s], k, tr[k - 1][r][s].x[k - 1] + inc)) any = 1;
                }
                if (!any) ok = 0;
            }
            if (ok && MODE && !common_alpha(k, NULL)) ok = 0;
            if (!ok) continue;
            ext = 1;
            chain[K++] = j; used[j] = 1;
            dfs();
            used[j] = 0; K--;
        }
    }
    if (!ext) {
        histmax[K]++;
        if (K > best) { best = K; nbest = 0; shown = 0; }
        if (K == best) { nbest++; if (shown < maxshow) { report(); shown++; } }
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); MODE = atoi(argv[2]); maxshow = argc > 3 ? atoi(argv[3]) : 2;
    bt(0, 0, 0, 0);
    if (argc > 4 && atoi(argv[4])) { /* excluir soluciones toroidales */
        int q = 0, nt = 0;
        for (int i = 0; i < Q; i++) {
            const int *s = sol + (long)i * N; int a[64] = {0}, b[64] = {0}, tor = 1;
            for (int r = 0; r < N; r++) { if (a[(s[r] + r) % N]++ || b[((s[r] - r) % N + N) % N]++) tor = 0; }
            if (tor) { nt++; continue; }
            memmove(sol + (long)q * N, s, sizeof(int) * N); q++;
        }
        printf("excluidas %d toroidales\n", nt); Q = q;
    }
    used = calloc(Q, 1);
    printf("N=%d soluciones=%d modo=%s\n", N, Q, MODE ? "alpha comun" : "alpha libre por reina");
    for (int i = 0; i < Q; i++) {
        chain[0] = i; K = 1; used[i] = 1;
        for (int r = 0; r < N; r++) for (int s = 0; s < 2; s++) {
            Track *t = &tr[0][r][s]; t->L = 0; t->U = N; t->ok = 1; t->x[0] = sol[(long)i * N + r];
        }
        dfs();
        used[i] = 0;
    }
    printf("  cadenas maximales por longitud:");
    for (int k = 1; k < MAXK; k++) if (histmax[k]) printf("  K=%d:%ld", k, histmax[k]);
    printf("\n  longitud maxima = %d  (%d cadenas)\n", best, nbest);
    return 0;
}
