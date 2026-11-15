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

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "uso: %s N K M R seed\n", argv[0]); return 1; }
    N = atoi(argv[1]); size_t K = atol(argv[2]); int M = atoi(argv[3]); int R = atoi(argv[4]);
    rng_s ^= (uint64_t)atol(argv[5]) * 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 10; i++) rnd();
    static const size_t counts[17] = {1,1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512};
    CNT = counts[N];
    S = calloc(CNT, 16);
    uint8_t q[16]; size_t pos = 0;
    gen(0, 0, 0, 0, q, &pos);
    printf("N=%d: %zu soluciones (esperadas %zu)\n", N, pos, CNT);
    fflush(stdout);

    // ---- nodos muestreados
    int all = (K >= CNT);
    if (all) K = CNT;
    uint32_t *qidx = malloc(K * sizeof(uint32_t));
    int *qseed = malloc(K * sizeof(int));
    for (size_t k = 0; k < K; k++) { qidx[k] = all ? (uint32_t)k : (uint32_t)(rnd() % CNT); qseed[k] = (int)k; }
    PV edges = {0};
    find_neighbors(qidx, qseed, K, &edges);
    int *deg = calloc(K, sizeof(int));
    long nref = 0, tot = 0; double two_cycles = 0, fixed_pts = 0;
    for (size_t e = 0; e < edges.n; e++) {
        int s = edges.v[e].seed; deg[s]++;
        const uint8_t *a = S + (size_t)qidx[s] * 16, *b = S + (size_t)edges.v[e].b * 16;
        int T[16];
        for (int i = 0; i < N; i++) T[a[i]] = b[i];
        int sft = T[0], isref = 1, c2 = 0, fx = 0;
        for (int c = 0; c < N; c++) {
            if (T[c] != ((sft - c) % N + N) % N) isref = 0;
            if (T[c] > c) c2++;
            if (T[c] == c) fx++;
        }
        nref += isref; tot++; two_cycles += c2; fixed_pts += fx;
    }
    // grado: media / mediana / min / max
    int *dd = malloc(K * sizeof(int)); memcpy(dd, deg, K * sizeof(int));
    for (size_t i = 0; i < K; i++) for (size_t j = i + 1; j < K && K <= 4000; j++) if (dd[j] < dd[i]) { int t = dd[i]; dd[i] = dd[j]; dd[j] = t; }
    double mean = (double)edges.n / K; int mn = dd[0], mx = dd[0];
    for (size_t i = 0; i < K; i++) { if (deg[i] < mn) mn = deg[i]; if (deg[i] > mx) mx = deg[i]; }
    printf("nodos muestreados=%zu  grado medio=%.2f  min=%d  max=%d%s\n", K, mean, mn, mx, all ? "  (todos los nodos)" : "");
    printf("aristas: reflexión modular c->s-c: %.1f%%   pares de columnas intercambiados por T: %.2f   puntos fijos de T: %.2f\n",
           100.0 * nref / (tot ? tot : 1), two_cycles / (tot ? tot : 1), fixed_pts / (tot ? tot : 1));
    fflush(stdout);

    // ---- bolas BFS
    if (M > (int)K) M = (int)K;
    if (M > 0 && R > 0) {
        typedef struct { uint32_t idx; uint8_t layer; } V;
        V **vis = malloc(M * sizeof(V *)); size_t *vn = calloc(M, sizeof(size_t));
        uint32_t **fr = malloc(M * sizeof(uint32_t *)); size_t *fn = calloc(M, sizeof(size_t));
        for (int s = 0; s < M; s++) {
            vis[s] = malloc(sizeof(V)); vis[s][0].idx = qidx[s]; vis[s][0].layer = 0; vn[s] = 1;
            fr[s] = malloc(sizeof(uint32_t)); fr[s][0] = qidx[s]; fn[s] = 1;
        }
        for (int layer = 1; layer <= R; layer++) {
            size_t nq = 0; for (int s = 0; s < M; s++) nq += fn[s];
            uint32_t *qi = malloc(nq * sizeof(uint32_t)); int *qs = malloc(nq * sizeof(int)); size_t p = 0;
            for (int s = 0; s < M; s++) for (size_t j = 0; j < fn[s]; j++) { qi[p] = fr[s][j]; qs[p] = s; p++; }
            PV out = {0};
            find_neighbors(qi, qs, nq, &out);
            qsort(out.v, out.n, sizeof(Pair), cmp_pair);
            for (int s = 0; s < M; s++) fn[s] = 0;
            size_t i = 0;
            while (i < out.n) {
                int s = out.v[i].seed; uint32_t b = out.v[i].b;
                size_t j = i; while (j < out.n && out.v[j].seed == s && out.v[j].b == b) j++;
                int seen = 0;                                        // ¿ya visitado?
                for (size_t t = 0; t < vn[s]; t++) if (vis[s][t].idx == b) { seen = 1; break; }
                if (!seen) {
                    vis[s] = realloc(vis[s], (vn[s] + 1) * sizeof(V));
                    vis[s][vn[s]].idx = b; vis[s][vn[s]].layer = (uint8_t)layer; vn[s]++;
                    fr[s] = realloc(fr[s], (fn[s] + 1) * sizeof(uint32_t)); fr[s][fn[s]++] = b;
                }
                i = j;
            }
            free(out.v); free(qi); free(qs);
            double avg = 0; for (int s = 0; s < M; s++) avg += vn[s]; avg /= M;
            printf("capa %d: tamaño medio de la bola (%d semillas) = %.1f   [consultas=%zu]\n", layer, M, avg, nq);
            fflush(stdout);
        }
        static const char *names[8] = {"id", "reflejo filas (A)", "reflejo cols (A)", "rot 180 (A)",
                                      "transpuesta (B)", "anti-transpuesta (B)", "rot 90 (B)", "rot 270 (B)"};
        int found[8] = {0}, firstlayer[8][8] = {{0}};
        for (int s = 0; s < M; s++) {
            uint8_t im[8][16]; images(S + (size_t)qidx[s] * 16, im);
            for (int t = 1; t < 8; t++) {
                long id = find_index(im[t]);
                for (size_t j = 0; j < vn[s]; j++) if ((long)vis[s][j].idx == id) { found[t]++; firstlayer[t][vis[s][j].layer]++; break; }
            }
        }
        printf("imágenes D4 de la semilla que caen dentro de la bola de radio %d (de %d semillas):\n", R, M);
        int fa = 0, fb = 0;
        for (int t = 1; t < 8; t++) {
            printf("   %-22s %4d", names[t], found[t]);
            printf("   (por capa:"); for (int l = 1; l <= R; l++) printf(" %d", firstlayer[t][l]); printf(")\n");
            if (t <= 3) fa += found[t]; else fb += found[t];
        }
        printf("   total grupo A (conserva mitad en N=8): %d de %d   |   grupo B (intercambia): %d de %d\n", fa, 3 * M, fb, 4 * M);
    }
    return 0;
}
