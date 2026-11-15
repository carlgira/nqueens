/*
 * cadenas.c — Patrón 1: cada reina es una onda leída por su fase, con su propia
 * velocidad entera v_r. En cada paso (cada π) la reina r avanza v_r columnas (mod N):
 *        c_r(k) = c_r(0) + k * v_r   (mod N)
 * Un grupo = (solución de partida S, vector de velocidades v). Buscamos cadenas en las
 * que S, S+v, S+2v, ... son todas soluciones (N-reinas ESTÁNDAR) y distintas.
 *
 * Truco: si S y S+v son soluciones, v = S' - S para alguna solución S'. Basta recorrer
 * los pares de soluciones (Q^2) en lugar de los N^N vectores.
 * El periodo de la cadena es p = N / gcd(N, v_0, ..., v_{N-1}); nunca hay más de p
 * tableros distintos. Si los p son soluciones, la cadena se cierra: un ciclo.
 *
 * uso: ./cadenas N [mostrar]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int N;
static int Q, cap;
static int *sol; /* Q x N */
static uint64_t *key;

static int cmpu(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static uint64_t enc(const int *c) { uint64_t k = 0; for (int r = 0; r < N; r++) k = k * N + c[r]; return k; }
static int is_sol_key(uint64_t k) { return bsearch(&k, key, Q, sizeof(uint64_t), cmpu) != NULL; }

static int cur[32];
static void bt(int r, unsigned cols, unsigned long long d1, unsigned long long d2) {
    if (r == N) {
        if (Q == cap) { cap = cap ? 2 * cap : 1024; sol = realloc(sol, sizeof(int) * N * cap); }
        memcpy(sol + (long)Q * N, cur, sizeof(int) * N); Q++; return;
    }
    for (int c = 0; c < N; c++) {
        if (cols >> c & 1) continue;
        if (d1 >> (r + c) & 1) continue;
        if (d2 >> (r - c + N) & 1) continue;
        cur[r] = c;
        bt(r + 1, cols | 1u << c, d1 | 1ull << (r + c), d2 | 1ull << (r - c + N));
    }
}
static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    int show = argc > 2 ? atoi(argv[2]) : 5;
    bt(0, 0, 0, 0);
    key = malloc(sizeof(uint64_t) * Q);
    for (int i = 0; i < Q; i++) key[i] = enc(sol + (long)i * N);
    qsort(key, Q, sizeof(uint64_t), cmpu);

    long hist[64] = {0}, hconst[64] = {0}, haff[64] = {0};
    int oS[6][32], oV[6][32], oL[6], nother = 0;      /* longitud de cadena (desde S, hacia delante) */
    long cycles[64] = {0};    /* ciclos cerrados por periodo */
    long cyc_const = 0, cyc_affine = 0, cyc_total = 0;
    int bestL = 0;
    int c[32], v[32];
    /* guardamos ejemplos de las mejores */
    int exS[8][32], exV[8][32], nex = 0;

    for (int i = 0; i < Q; i++) {
        const int *S = sol + (long)i * N;
        for (int j = 0; j < Q; j++) {
            if (i == j) continue;
            const int *T = sol + (long)j * N;
            int g = N;
            for (int r = 0; r < N; r++) { v[r] = ((T[r] - S[r]) % N + N) % N; g = gcd(g, v[r]); }
            int p = N / g;
            /* ¿S es el comienzo de la cadena? (S - v no es solución) -> contamos cadenas maximales */
            for (int r = 0; r < N; r++) c[r] = ((S[r] - v[r]) % N + N) % N;
            int prev_ok = is_sol_key(enc(c));
            int L = 2; /* S y T */
            for (int k = 2; k < p; k++) {
                for (int r = 0; r < N; r++) c[r] = (S[r] + k * v[r]) % N;
                if (!is_sol_key(enc(c))) break;
                L++;
            }
            if (L == p) {
                /* ciclo cerrado: lo contamos una vez por (ciclo, dirección) -> cada ciclo aparece
                   p veces (cada punto de partida) en esta dirección */
                cycles[p]++;
                cyc_total++;
                int isconst = 1; for (int r = 1; r < N; r++) if (v[r] != v[0]) isconst = 0;
                int isaff = 1; int a = ((v[1] - v[0]) % N + N) % N;
                for (int r = 2; r < N; r++) if (((v[r] - v[r - 1]) % N + N) % N != a) isaff = 0;
                cyc_const += isconst; cyc_affine += isaff;
            } else if (!prev_ok) {
                hist[L]++;
                int isconst = 1; for (int r = 1; r < N; r++) if (v[r] != v[0]) isconst = 0;
                int isaff = 1; int a = ((v[1] - v[0]) % N + N) % N;
                for (int r = 2; r < N; r++) if (((v[r] - v[r - 1]) % N + N) % N != a) isaff = 0;
                if (isconst) hconst[L]++; else if (isaff) haff[L]++;
                if (!isaff && L >= 3 && nother < 6) { memcpy(oS[nother], S, sizeof(int)*N); memcpy(oV[nother], v, sizeof(int)*N); oL[nother]=L; nother++; }
            }
            if (L > bestL) { bestL = L; nex = 0; }
            if (L == bestL && nex < 8) { memcpy(exS[nex], S, sizeof(int) * N); memcpy(exV[nex], v, sizeof(int) * N); nex++; }
        }
    }
    printf("N=%d soluciones=%d  pares=%ld\n", N, Q, (long)Q * (Q - 1));
    printf("  cadenas abiertas maximales por longitud:");
    for (int L = 2; L < 64; L++) if (hist[L]) printf("  L=%d:%ld", L, hist[L]);
    printf("\n  por longitud (L>=3): constante / afin no constante / otros:");
    for (int L = 3; L < 64; L++) if (hist[L]) printf("  L=%d: %ld/%ld/%ld", L, hconst[L], haff[L], hist[L]-hconst[L]-haff[L]);
    for (int e = 0; e < nother; e++) { printf("\n   otro L=%d S=", oL[e]); for (int r=0;r<N;r++) printf("%d%s",oS[e][r],r<N-1?",":""); printf(" v="); for (int r=0;r<N;r++) printf("%d%s",oV[e][r],r<N-1?",":""); }
    printf("\n  ciclos cerrados (contados por punto de partida y sentido):");
    for (int p = 2; p < 64; p++) if (cycles[p]) printf("  p=%d:%ld (=%ld ciclos)", p, cycles[p], cycles[p] / p / 2);
    printf("\n  de ellos: v constante %ld, v afin en la fila %ld, otros %ld\n", cyc_const, cyc_affine, cyc_total - cyc_affine);
    printf("  longitud maxima = %d\n", bestL);
    for (int e = 0; e < nex && e < show; e++) {
        printf("   S=");
        for (int r = 0; r < N; r++) printf("%d%s", exS[e][r], r < N - 1 ? "," : "");
        printf("  v=");
        for (int r = 0; r < N; r++) printf("%d%s", exV[e][r], r < N - 1 ? "," : "");
        printf("\n");
    }
    return 0;
}
