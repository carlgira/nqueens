/*
 * bucles.c — Censo de bucles: N órbitas con un planeta cada una, velocidad constante entera
 * v_r (sectores por foto). Desde una condición inicial c, las fotos son c + k v (mod N) y
 * forman un bucle de periodo <= N. El "grupo" de esa condición inicial es el conjunto de
 * soluciones que toca el bucle (versión relajada: entre medias puede haber fotos malas).
 *
 * Enumera todos los bucles que tocan >= 2 tableros del conjunto: para cada par (S,T) y cada
 * k divisor propio de N (1, 2, 4 para N=8) resuelve k v = T - S (mod N).
 * Escribe un grupo por línea (índices ordenados); deduplicar con sort -u.
 *
 * uso: ./bucles N conjunto(0 sol, s<0 simétrico, s>0 azar) | sort -u
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int N, Q, cap; static int *sol; static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) { if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return; }
    for (int c = 0; c < N; c++) { if (cols >> c & 1 || d1 >> (r + c) & 1 || d2 >> (r - c + N) & 1) continue; cur[r] = c; bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N)); }
}
static unsigned long long rs = 12345;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static void randperm(int *p) { for (int r = 0; r < N; r++) p[r] = r; for (int r = N - 1; r > 0; r--) { int j = (int)(ur() * (r + 1)); int t = p[r]; p[r] = p[j]; p[j] = t; } }
static int have(const int *g, int n) { for (int k = 0; k < n; k++) if (!memcmp(g, sol + (long)k * N, sizeof(int) * N)) return 1; return 0; }
static void push(const int *g) { if (Q == cap) { cap = cap ? cap * 2 : 1024; sol = realloc(sol, sizeof(int) * N * cap); } memcpy(sol + (long)Q * N, g, sizeof(int) * N); Q++; }
static int *idx; /* tablero (base N) -> índice o -1 */
static long key(const int *c) { long k = 0; for (int r = 0; r < N; r++) k = k * N + c[r]; return k; }
static int cmpi(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

static void loop(const int *S, const int *v) {
    int g[64], n = 0, c[32];
    for (int k = 0; k < N; k++) {
        for (int r = 0; r < N; r++) c[r] = (S[r] + k * v[r]) % N;
        int i = idx[key(c)]; if (i < 0) continue;
        int dup = 0; for (int a = 0; a < n; a++) if (g[a] == i) dup = 1;
        if (!dup) g[n++] = i;
    }
    if (n < 2) return;
    qsort(g, n, sizeof(int), cmpi);
    printf("%d |", n); for (int a = 0; a < n; a++) printf(" %d", g[a]); printf("\n");
}
static int v[32], base[32], kk;
static void rec(const int *S, int r) {
    if (r == N) { loop(S, v); return; }
    for (int e = 0; e < kk; e++) { v[r] = (base[r] + e * (N / kk)) % N; rec(S, r + 1); }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); int set = atoi(argv[2]);
    bt(0, 0, 0, 0); int want = Q;
    if (set > 0) { rs ^= (unsigned long long)set * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32]; while (Q < want) { randperm(p); if (!have(p, Q)) push(p); } }
    else if (set < 0) {
        rs ^= (unsigned long long)(-set) * 0x9E3779B97F4A7C15ULL; Q = 0; int p[32], g[32], q[32];
        while (Q < want) { randperm(p); for (int e = 0; e < 8; e++) { memcpy(q, p, sizeof q); if (e & 4) for (int r = 0; r < N; r++) q[p[r]] = r; for (int r = 0; r < N; r++) g[r] = q[(e & 2) ? N - 1 - r : r]; if (e & 1) for (int r = 0; r < N; r++) g[r] = N - 1 - g[r]; if (!have(g, Q)) push(g); } }
    }
    long tot = 1; for (int r = 0; r < N; r++) tot *= N;
    idx = malloc(sizeof(int) * tot); for (long i = 0; i < tot; i++) idx[i] = -1;
    for (int i = 0; i < Q; i++) idx[key(sol + (long)i * N)] = i;
    for (int i = 0; i < Q; i++) for (int j = 0; j < Q; j++) {
        if (i == j) continue;
        const int *S = sol + (long)i * N, *T = sol + (long)j * N;
        for (kk = 1; kk < N; kk++) {
            if (N % kk) continue;
            int ok = 1;
            for (int r = 0; r < N; r++) { int d = ((T[r] - S[r]) % N + N) % N; if (d % kk) { ok = 0; break; } base[r] = d / kk; }
            if (ok) rec(S, 0);
        }
    }
    return 0;
}
