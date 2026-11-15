// Cayley <= 2 por tipo de arista, con componentes EXACTAS (union-find sobre todas las soluciones), N <= 16.
//
// Vecinos de una solución q (q[i] = columna de la reina en la fila i), generados directamente:
//   tipo 1 "intercambio"      : T = una transposición de columnas          (Cayley 1)  -> C(N,2)  candidatos
//   tipo 2 "ciclo de 3"       : T = ciclo de longitud 3                     (Cayley 2)  -> 2*C(N,3)
//   tipo 3 "doble intercambio": T = dos transposiciones disjuntas           (Cayley 2)  -> 3*C(N,4)
// Un candidato es vecino si el tablero resultante sigue siendo solución (se comprueban las diagonales
// solo de las filas que cambian). Reglas del grafo:  sin aristas entre imágenes D4  y  sin toroidal-toroidal.
// Cada arista no dirigida se cuenta una vez (índice del vecino > índice de la solución).
//
// Uso: ./nq_cayley N
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

static int N;
static uint8_t *S;
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

static long find_index(const uint8_t *key) {
    size_t lo = 0, hi = CNT;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        int c = memcmp(S + mid * 16, key, N);
        if (c == 0) return (long)mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return -1;
}

static int is_tor(const uint8_t *q) {                 // diagonales rotas también distintas
    uint32_t m1 = 0, m2 = 0, full = (1u << N) - 1;
    for (int i = 0; i < N; i++) { m1 |= 1u << ((q[i] + i) % N); m2 |= 1u << ((q[i] - i + N) % N); }
    return m1 == full && m2 == full;
}

// ---- union-find
typedef struct { int32_t *p; } UF;
static void uf_init(UF *u) { u->p = malloc(CNT * sizeof(int32_t)); for (size_t i = 0; i < CNT; i++) u->p[i] = (int32_t)i; }
static int32_t uf_find(UF *u, int32_t x) { while (u->p[x] != x) { u->p[x] = u->p[u->p[x]]; x = u->p[x]; } return x; }
static void uf_union(UF *u, int32_t a, int32_t b) { int32_t ra = uf_find(u, a), rb = uf_find(u, b); if (ra != rb) u->p[ra] = rb; }

static UF UFS[4];                 // 0 = todos los tipos, 1 = intercambio, 2 = ciclo de 3, 3 = doble intercambio
static long long edges[4], excl_sym, excl_tor;

static inline int valid(const int *rows, const int *nc, int k, uint32_t m1, uint32_t m2, const int *d1, const int *d2) {
    uint32_t r1 = m1, r2 = m2, a1 = 0, a2 = 0;
    for (int t = 0; t < k; t++) { r1 &= ~(1u << d1[rows[t]]); r2 &= ~(1u << d2[rows[t]]); }
    for (int t = 0; t < k; t++) {
        uint32_t b1 = 1u << (rows[t] - nc[t] + N - 1), b2 = 1u << (rows[t] + nc[t]);
        if (((r1 | a1) & b1) || ((r2 | a2) & b2)) return 0;
        a1 |= b1; a2 |= b2;
    }
    return 1;
}

static inline void found(size_t a, const uint8_t *q, int type, const int *rows, const int *nc, int k) {
    uint8_t tmp[16]; memcpy(tmp, q, 16);
    for (int t = 0; t < k; t++) tmp[rows[t]] = (uint8_t)nc[t];
    long b = find_index(tmp);
    if (b < 0 || (size_t)b <= a) return;                       // cada arista una vez
    uint8_t im[8][16]; images(q, im);
    for (int s = 0; s < 8; s++) if (memcmp(im[s], tmp, N) == 0) {
        #pragma omp atomic
        excl_sym++;
        return;
    }
    if (is_tor(q) && is_tor(tmp)) {
        #pragma omp atomic
        excl_tor++;
        return;
    }
    #pragma omp critical
    {
        uf_union(&UFS[0], (int32_t)a, (int32_t)b);
        uf_union(&UFS[type], (int32_t)a, (int32_t)b);
        edges[type]++;
    }
}

static int cmp_desc(const void *x, const void *y) { int a = *(const int *)x, b = *(const int *)y; return b - a; }

static void report(const char *name, UF *u, long long ne) {
    int32_t *cnt = calloc(CNT, sizeof(int32_t));
    for (size_t i = 0; i < CNT; i++) cnt[uf_find(u, (int32_t)i)]++;
    size_t comps = 0, nontriv = 0, iso = 0; int mx = 0;
    int *sizes = malloc(CNT * sizeof(int)); size_t ns = 0;
    for (size_t i = 0; i < CNT; i++) if (cnt[i]) { comps++; if (cnt[i] > 1) { nontriv++; sizes[ns++] = cnt[i]; } else iso++; if (cnt[i] > mx) mx = cnt[i]; }
    qsort(sizes, ns, sizeof(int), cmp_desc);
    printf("  %-18s aristas=%10lld  grado medio=%6.2f  comps=%8zu  no triviales=%7zu  aislados=%9zu  mayor=%9d (%5.1f%%)  top=[",
           name, ne, 2.0 * ne / CNT, comps, nontriv, iso, mx, 100.0 * mx / CNT);
    for (size_t i = 0; i < ns && i < 6; i++) printf("%s%d", i ? ", " : "", sizes[i]);
    printf("]\n");
    free(cnt); free(sizes);
}

static int odd_perm(const uint8_t *q) {            // 1 si q (como permutación) es impar
    int seen[16] = {0}, cycles = 0;
    for (int i = 0; i < N; i++) if (!seen[i]) { cycles++; int j = i; while (!seen[j]) { seen[j] = 1; j = q[j]; } }
    return (N - cycles) & 1;
}

static void check_parity(const char *name, UF *u) {         // ¿alguna componente mezcla pares e impares?
    int8_t *rp = malloc(CNT); memset(rp, -1, CNT);
    size_t mixed = 0, mixed_nodes = 0;
    int8_t *bad = calloc(CNT, 1);
    for (size_t i = 0; i < CNT; i++) {
        int32_t r = uf_find(u, (int32_t)i); int p = odd_perm(S + i * 16);
        if (rp[r] < 0) rp[r] = (int8_t)p; else if (rp[r] != p && !bad[r]) { bad[r] = 1; mixed++; }
    }
    for (size_t i = 0; i < CNT; i++) if (bad[uf_find(u, (int32_t)i)]) mixed_nodes++;
    printf("  paridad de q en %-18s componentes que mezclan pares e impares: %zu  (%zu soluciones en ellas)\n", name, mixed, mixed_nodes);
    free(rp); free(bad);
}

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    if (argc > 2 && strcmp(argv[2], "parity") == 0) {         // modo solo paridad: cuántas soluciones pares/impares
        static const size_t cc[17] = {1,1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512};
        CNT = cc[N]; S = calloc(CNT, 16);
        uint8_t qq[16]; size_t pp = 0; gen(0, 0, 0, 0, qq, &pp);
        size_t odd = 0; for (size_t i = 0; i < CNT; i++) odd += odd_perm(S + i * 16);
        printf("N=%d: soluciones con q par=%zu, impar=%zu\n", N, CNT - odd, odd);
        return 0;
    }
    static const size_t counts[17] = {1,1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512};
    CNT = counts[N]; S = calloc(CNT, 16);
    uint8_t q0[16]; size_t pos = 0; gen(0, 0, 0, 0, q0, &pos);
    for (int t = 0; t < 4; t++) uf_init(&UFS[t]);
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t a = 0; a < CNT; a++) {
        const uint8_t *q = S + a * 16;
        int d1[16], d2[16]; uint32_t m1 = 0, m2 = 0;
        for (int i = 0; i < N; i++) { d1[i] = i - q[i] + N - 1; d2[i] = i + q[i]; m1 |= 1u << d1[i]; m2 |= 1u << d2[i]; }
        int rows[4], nc[4];
        for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
            rows[0] = i; rows[1] = j; nc[0] = q[j]; nc[1] = q[i];
            if (valid(rows, nc, 2, m1, m2, d1, d2)) found(a, q, 1, rows, nc, 2);
            for (int k = j + 1; k < N; k++) {
                rows[0] = i; rows[1] = j; rows[2] = k;
                nc[0] = q[j]; nc[1] = q[k]; nc[2] = q[i];
                if (valid(rows, nc, 3, m1, m2, d1, d2)) found(a, q, 2, rows, nc, 3);
                nc[0] = q[k]; nc[1] = q[i]; nc[2] = q[j];
                if (valid(rows, nc, 3, m1, m2, d1, d2)) found(a, q, 2, rows, nc, 3);
                for (int l = k + 1; l < N; l++) {
                    rows[0] = i; rows[1] = j; rows[2] = k; rows[3] = l;
                    nc[0] = q[j]; nc[1] = q[i]; nc[2] = q[l]; nc[3] = q[k];       // (ij)(kl)
                    if (valid(rows, nc, 4, m1, m2, d1, d2)) found(a, q, 3, rows, nc, 4);
                    nc[0] = q[k]; nc[1] = q[l]; nc[2] = q[i]; nc[3] = q[j];       // (ik)(jl)
                    if (valid(rows, nc, 4, m1, m2, d1, d2)) found(a, q, 3, rows, nc, 4);
                    nc[0] = q[l]; nc[1] = q[k]; nc[2] = q[j]; nc[3] = q[i];       // (il)(jk)
                    if (valid(rows, nc, 4, m1, m2, d1, d2)) found(a, q, 3, rows, nc, 4);
                }
            }
        }
    }
    long long tot = edges[1] + edges[2] + edges[3];
    long tor = 0; for (size_t i = 0; i < CNT; i++) tor += is_tor(S + i * 16);
    printf("=== N=%d: %zu soluciones (%ld toroidales)  [%.0f s]   aristas excluidas: simetría D4=%lld, toroidal-toroidal=%lld\n",
           N, CNT, tor, omp_get_wtime() - t0, excl_sym, excl_tor);
    report("Cayley<=2 (todas)", &UFS[0], tot);
    report("solo intercambio", &UFS[1], edges[1]);
    report("solo ciclo de 3", &UFS[2], edges[2]);
    report("solo doble interc.", &UFS[3], edges[3]);
    check_parity("solo intercambio", &UFS[1]);
    check_parity("solo ciclo de 3", &UFS[2]);
    check_parity("solo doble interc.", &UFS[3]);
    {
        size_t odd = 0; for (size_t i = 0; i < CNT; i++) odd += odd_perm(S + i * 16);
        printf("  soluciones con q par=%zu, impar=%zu\n", CNT - odd, odd);
    }
    if (argc > 2 && strcmp(argv[2], "query") == 0) {          // soluciones dadas por stdin (columnas 1..N por fila)
        static const char *nm[4] = {"todas", "interc", "ciclo3", "doble"};
        int32_t *cnt[4]; int32_t mx[4];
        for (int t = 0; t < 4; t++) {
            cnt[t] = calloc(CNT, sizeof(int32_t)); mx[t] = 0;
            for (size_t i = 0; i < CNT; i++) cnt[t][uf_find(&UFS[t], (int32_t)i)]++;
            for (size_t i = 0; i < CNT; i++) if (cnt[t][i] > mx[t]) mx[t] = cnt[t][i];
        }
        char line[512];
        while (fgets(line, sizeof line, stdin)) {
            uint8_t q[16] = {0}; int k = 0; char *p = line;
            while (k < N) { char *e; long v = strtol(p, &e, 10); if (e == p) break; q[k++] = (uint8_t)(v - 1); p = e; while (*p == ',' || *p == ' ') p++; }
            if (k < N) continue;
            long idx = find_index(q);
            if (idx < 0) { printf("  consulta no es solución válida\n"); continue; }
            printf("  consulta (%s): %s\n    ", odd_perm(q) ? "impar" : "par", strtok(line, "\n"));
            for (int t = 0; t < 4; t++) {
                int32_t sz = cnt[t][uf_find(&UFS[t], (int32_t)idx)];
                printf("%s: comp=%d (mayor=%d)%s   ", nm[t], sz, mx[t], sz == mx[t] ? " [GIGANTE]" : (sz * 10 >= mx[t] * 8 ? " [~gigante]" : ""));
            }
            printf("\n");
        }
    }
    return 0;
}
