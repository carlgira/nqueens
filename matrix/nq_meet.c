// Muestra local del grafo de involución de N-reinas (N <= 16), sin construir el grafo completo.
//
// Arista a--b  <=>  T = P_b P_a^T es involución (T[a[i]] = b[i] y T[b[i]] = a[i]), b no es imagen D4 de a.
// Test rápido: para todo i, b[ia[b[i]]] == a[i]   (ia = inversa de a).
//
// Uso: ./nq_sample N K M R seed
//   K = nº de nodos muestreados para estadísticas de grado (si K >= nº de soluciones se usan todos)
//   M = nº de semillas (de las K) para la bola BFS de radio R
//   R = radio de la bola
// Salida: grado, tipo de T, tamaño de la bola, y qué imágenes D4 de la semilla caen dentro de la bola.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

static int N;
static uint8_t *S;          // soluciones, 16 bytes por solución (relleno), orden lexicográfico
static size_t CNT;

static void gen(int r, uint32_t cols, uint32_t d1, uint32_t d2, uint8_t *q, size_t *pos) {
    uint32_t full = (1u << N) - 1;
    if (r == N) { memcpy(S + *pos * 16, q, N); (*pos)++; return; }
    uint32_t av = ~(cols | d1 | d2) & full;
    while (av) {
        uint32_t bit = av & -av; av ^= bit;
        q[r] = (uint8_t)__builtin_ctz(bit);
        gen(r + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1, q, pos);
    }
}

static void images(const uint8_t *q, uint8_t out[8][16]) {
    uint8_t inv[16];
    for (int i = 0; i < N; i++) inv[q[i]] = (uint8_t)i;
    memset(out, 0, 8 * 16);
    for (int i = 0; i < N; i++) {
        out[0][i] = q[i];
        out[1][i] = q[N - 1 - i];
        out[2][i] = (uint8_t)(N - 1 - q[i]);
        out[3][i] = (uint8_t)(N - 1 - q[N - 1 - i]);
        out[4][i] = inv[i];
        out[5][i] = inv[N - 1 - i];
        out[6][i] = (uint8_t)(N - 1 - inv[i]);
        out[7][i] = (uint8_t)(N - 1 - inv[N - 1 - i]);
    }
}

static long find_index(const uint8_t *key) {           // búsqueda binaria lexicográfica
    size_t lo = 0, hi = CNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        int c = memcmp(S + mid * 16, key, N);
        if (c == 0) return (long)mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return -1;
}

typedef struct { int seed; uint32_t b; } Pair;
typedef struct { Pair *v; size_t n, cap; } PV;
static void pv_push(PV *p, int s, uint32_t b) {
    if (p->n == p->cap) { p->cap = p->cap ? p->cap * 2 : 1024; p->v = realloc(p->v, p->cap * sizeof(Pair)); }
    p->v[p->n].seed = s; p->v[p->n].b = b; p->n++;
}
static int cmp_pair(const void *x, const void *y) {
    const Pair *a = x, *b = y;
    if (a->seed != b->seed) return a->seed < b->seed ? -1 : 1;
    if (a->b != b->b) return a->b < b->b ? -1 : 1;
    return 0;
}

// vecinos (por involución, sin imágenes D4) de cada consulta; devuelve pares (semilla, índice b)
static void find_neighbors(const uint32_t *qidx, const int *qseed, size_t nq, PV *out) {
    uint8_t (*A)[16] = malloc(nq * 16), (*IA)[16] = malloc(nq * 16);
    uint8_t (*IM)[8][16] = malloc(nq * 8 * 16);
    for (size_t k = 0; k < nq; k++) {
        const uint8_t *a = S + (size_t)qidx[k] * 16;
        memcpy(A[k], a, 16);
        for (int i = 0; i < N; i++) IA[k][a[i]] = (uint8_t)i;
        images(a, IM[k]);
    }
    int nth = omp_get_max_threads();
    PV *loc = calloc(nth, sizeof(PV));
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t b = 0; b < CNT; b++) {
            const uint8_t *bb = S + b * 16;
            for (size_t k = 0; k < nq; k++) {
                const uint8_t *a = A[k], *ia = IA[k];
                int ok = 1;
                for (int i = 0; i < N; i++) if (bb[ia[bb[i]]] != a[i]) { ok = 0; break; }
                if (!ok) continue;
                int sym = 0;
                for (int s = 0; s < 8; s++) if (memcmp(IM[k][s], bb, N) == 0) { sym = 1; break; }
                if (!sym) pv_push(&loc[t], qseed[k], (uint32_t)b);
            }
        }
    }
    for (int t = 0; t < nth; t++) {
        for (size_t i = 0; i < loc[t].n; i++) pv_push(out, loc[t].v[i].seed, loc[t].v[i].b);
        free(loc[t].v);
    }
    free(loc); free(A); free(IA); free(IM);
}

static uint64_t rng_s = 88172645463325252ULL;
static uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static int cmpu(const void *x, const void *y) { uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y; return a < b ? -1 : a > b; }

// Prueba "encuentro en el medio": bolas de radio R de a y de algunas de sus imágenes D4; ¿se tocan?
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s N M R seed\n", argv[0]); return 1; }
    N = atoi(argv[1]); int M = atoi(argv[2]); int R = atoi(argv[3]);
    rng_s ^= (uint64_t)atol(argv[4]) * 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 10; i++) rnd();
    static const size_t counts[17] = {1,1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512};
    CNT = counts[N]; S = calloc(CNT, 16);
    uint8_t q[16]; size_t pos = 0; gen(0, 0, 0, 0, q, &pos);
    printf("N=%d: %zu soluciones\n", N, pos); fflush(stdout);
    static const int IMG[5] = {0, 1, 3, 4, 6};
    static const char *NAME[5] = {"id", "reflejo filas (A)", "rot 180 (A)", "transpuesta (B)", "rot 90 (B)"};
    int NR = 5, NS = M * NR;
    uint32_t *roots = malloc(NS * sizeof(uint32_t));
    for (int m = 0; m < M; m++) {
        uint32_t idx = (uint32_t)(rnd() % CNT);
        uint8_t im[8][16]; images(S + (size_t)idx * 16, im);
        for (int j = 0; j < NR; j++) roots[m * NR + j] = (uint32_t)find_index(im[IMG[j]]);
    }
    size_t bmsz = CNT / 8 + 1;
    uint8_t **bm = malloc(NS * sizeof(uint8_t *));
    typedef struct { uint32_t idx; uint8_t layer; } V;
    V **vis = malloc(NS * sizeof(V *)); size_t *vn = calloc(NS, sizeof(size_t)), *vc = calloc(NS, sizeof(size_t));
    uint32_t **fr = malloc(NS * sizeof(uint32_t *)); size_t *fn = calloc(NS, sizeof(size_t)), *fc = calloc(NS, sizeof(size_t));
    for (int s = 0; s < NS; s++) {
        bm[s] = calloc(bmsz, 1); bm[s][roots[s] >> 3] |= 1 << (roots[s] & 7);
        vc[s] = 1024; vis[s] = malloc(vc[s] * sizeof(V)); vis[s][0].idx = roots[s]; vis[s][0].layer = 0; vn[s] = 1;
        fc[s] = 1024; fr[s] = malloc(fc[s] * sizeof(uint32_t)); fr[s][0] = roots[s]; fn[s] = 1;
    }
    for (int layer = 1; layer <= R; layer++) {
        size_t nq = 0; for (int s = 0; s < NS; s++) nq += fn[s];
        uint32_t *qi = malloc(nq * sizeof(uint32_t)); int *qs = malloc(nq * sizeof(int)); size_t p = 0;
        for (int s = 0; s < NS; s++) for (size_t j = 0; j < fn[s]; j++) { qi[p] = fr[s][j]; qs[p] = s; p++; }
        PV out = {0}; find_neighbors(qi, qs, nq, &out);
        for (int s = 0; s < NS; s++) fn[s] = 0;
        for (size_t e = 0; e < out.n; e++) {
            int s = out.v[e].seed; uint32_t b = out.v[e].b;
            if (bm[s][b >> 3] & (1 << (b & 7))) continue;
            bm[s][b >> 3] |= 1 << (b & 7);
            if (vn[s] == vc[s]) { vc[s] *= 2; vis[s] = realloc(vis[s], vc[s] * sizeof(V)); }
            vis[s][vn[s]].idx = b; vis[s][vn[s]].layer = (uint8_t)layer; vn[s]++;
            if (fn[s] == fc[s]) { fc[s] *= 2; fr[s] = realloc(fr[s], fc[s] * sizeof(uint32_t)); }
            fr[s][fn[s]++] = b;
        }
        free(out.v); free(qi); free(qs);
        printf("capa %d completada (%zu consultas)\n", layer, nq); fflush(stdout);
    }
    int touch[5] = {0}; double ovl[5] = {0}; int mind[5]; for (int j = 0; j < 5; j++) mind[j] = 99;
    for (int m = 0; m < M; m++) {
        int sa = m * NR;
        printf("semilla %d: bola(a)=%zu", m, vn[sa]);
        for (int j = 1; j < NR; j++) {
            int sb = m * NR + j; size_t common = 0; int best = 99;
            for (size_t t = 0; t < vn[sb]; t++) {
                uint32_t b = vis[sb][t].idx;
                if (bm[sa][b >> 3] & (1 << (b & 7))) {
                    common++;
                    // distancia a través de este nodo: capa en a + capa en la imagen
                    for (size_t u = 0; u < vn[sa]; u++) if (vis[sa][u].idx == b) { int d = vis[sa][u].layer + vis[sb][t].layer; if (d < best) best = d; break; }
                }
            }
            printf("  | %s: bola=%zu comunes=%zu%s", NAME[j], vn[sb], common, common ? "" : " (0)");
            if (common) { touch[j]++; ovl[j] += common; if (best < mind[j]) mind[j] = best; }
        }
        printf("\n");
    }
    printf("\nresumen (%d semillas, bolas de radio %d): ¿la bola de a toca la bola de su imagen?\n", M, R);
    for (int j = 1; j < NR; j++)
        printf("   %-20s tocan en %d de %d  (nodos comunes medios=%.1f%s)\n", NAME[j], touch[j], M, ovl[j] / M,
               touch[j] ? "" : "");
    return 0;
}
