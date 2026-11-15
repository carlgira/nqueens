// salto_bits.c — caminos entre soluciones construidos con bits, pasando por tableros no válidos.
//
// Tablero = máscara de N² bits (N reinas). Para ir de la solución A a la B:
//   sobran  = A AND NOT B      (reinas que hay que quitar)
//   faltan  = B AND NOT A      (casillas que hay que llenar)
//   cambia  = A XOR B
// Se empareja cada reina que sobra con una casilla que falta (el emparejamiento
// de menor distancia de rey total) y se mueven de una en una, un salto de rey
// cada vez, hacia su destino. Los tableros intermedios no tienen por qué ser
// válidos. No se busca nada: el camino se construye.
//
// Secuencias: desde una solución al azar se va a otra al azar no visitada, y
// así hasta pasar por todas. Las parejas ya hechas se reutilizan, no se
// rehacen. Todos los caminos juntos forman un grafo del que se sacan
// estadísticas.
//
// Uso: ./salto_bits N secuencias [semilla]     (N ≤ 8)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N, NC;
static uint64_t S[128]; static int ns, xs[8];
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int iabs(int a) { return a < 0 ? -a : a; }
static int imax(int a, int b) { return a > b ? a : b; }

static void bt(int c, int rows, int d1, int d2) {
    if (c == N) { uint64_t m = 0; for (int i = 0; i < N; i++) m |= 1ULL << (xs[i] * N + i); S[ns++] = m; return; }
    for (int r = 0; r < N; r++) {
        if ((rows >> r & 1) || (d1 >> (r + c) & 1) || (d2 >> (r - c + N) & 1)) continue;
        xs[c] = r; bt(c + 1, rows | 1 << r, d1 | 1 << (r + c), d2 | 1 << (r - c + N));
    }
}
static int sol_index(uint64_t m) { for (int k = 0; k < ns; k++) if (S[k] == m) return k; return -1; }
static int conflicts(uint64_t m) {                // pares de reinas que se atacan
    int p[64], k = 0, c = 0;
    for (int i = 0; i < NC; i++) if (m >> i & 1) p[k++] = i;
    for (int a = 0; a < k; a++) for (int b = a + 1; b < k; b++) {
        int r1 = p[a] / N, c1 = p[a] % N, r2 = p[b] / N, c2 = p[b] % N;
        if (r1 == r2 || c1 == c2 || iabs(r1 - r2) == iabs(c1 - c2)) c++;
    }
    return c;
}

// ---------- camino guiado por bits de A a B ----------
static uint64_t path[4096]; static int plen;
static int cheb(int p, int q) { return imax(iabs(p / N - q / N), iabs(p % N - q % N)); }
static int bestperm[8], perm[8], fr[8], to[8], nk, bestcost;
static void assign(int i, int used, int cost) {   // emparejamiento de menor distancia total (k ≤ 8)
    if (cost >= bestcost) return;
    if (i == nk) { bestcost = cost; memcpy(bestperm, perm, sizeof perm); return; }
    for (int j = 0; j < nk; j++) if (!(used >> j & 1)) { perm[i] = j; assign(i + 1, used | 1 << j, cost + cheb(fr[i], to[j])); }
}
static int build_path(uint64_t A, uint64_t B, int *optimal_len) {
    uint64_t sob = A & ~B, fal = B & ~A;
    nk = 0; for (int i = 0; i < NC; i++) if (sob >> i & 1) fr[nk++] = i;
    int m = 0; for (int i = 0; i < NC; i++) if (fal >> i & 1) to[m++] = i;
    bestcost = 1 << 30; assign(0, 0, 0); *optimal_len = bestcost;
    int pos[8], tgt[8]; for (int i = 0; i < nk; i++) { pos[i] = fr[i]; tgt[i] = to[bestperm[i]]; }
    uint64_t cur = A; plen = 0; path[plen++] = cur;
    int guard = 0;
    while (cur != B && guard++ < 4000) {
        // reina al azar que no ha llegado; un salto de rey hacia su destino si la casilla está libre
        int cand[8], nc = 0; for (int i = 0; i < nk; i++) if (pos[i] != tgt[i]) cand[nc++] = i;
        int moved = 0;
        for (int tries = 0; tries < nc && !moved; tries++) {
            int i = cand[(rnd() + tries) % nc];
            int r = pos[i] / N, c = pos[i] % N, tr = tgt[i] / N, tc = tgt[i] % N;
            int r2 = r + (tr > r) - (tr < r), c2 = c + (tc > c) - (tc < c), q = r2 * N + c2;
            if (cur >> q & 1) continue;
            cur = (cur & ~(1ULL << pos[i])) | (1ULL << q); pos[i] = q; moved = 1;
        }
        if (!moved) {                               // todas bloqueadas: salto lateral al azar
            int i = cand[rnd() % nc], r = pos[i] / N, c = pos[i] % N;
            for (int t = 0; t < 16; t++) { int r2 = r + (int)(rnd() % 3) - 1, c2 = c + (int)(rnd() % 3) - 1;
                if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue; int q = r2 * N + c2; if (cur >> q & 1) continue;
                cur = (cur & ~(1ULL << pos[i])) | (1ULL << q); pos[i] = q; break; }
        }
        path[plen++] = cur;
    }
    return plen - 1;
}

// ---------- grafo unión: tablero -> nº de caminos de pareja que lo usan ----------
typedef struct { uint64_t key; int uses; } Node;
static Node *G; static uint64_t gcap; static long gnodes;
static uint64_t h64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static Node *gget(uint64_t k) {
    uint64_t h = h64(k) & (gcap - 1);
    while (G[h].key && G[h].key != k) h = (h + 1) & (gcap - 1);
    if (!G[h].key) { G[h].key = k; G[h].uses = 0; gnodes++; }
    return &G[h];
}

static int cmpl(const void *x, const void *y) { long a = *(const long *)x, b = *(const long *)y; return a < b ? -1 : a > b; }

int main(int argc, char **argv) {
    N = atoi(argv[1]); NC = N * N; int R = argc > 2 ? atoi(argv[2]) : 1000;
    if (argc > 3) rs ^= (uint64_t)atoll(argv[3]) * 0x9E3779B97F4A7C15ULL;
    bt(0, 0, 0, 0);
    gcap = 1 << 22; G = calloc(gcap, sizeof(Node));
    printf("N = %d   soluciones %d\n", N, ns);

    // 1) un camino por cada pareja (A < B), con los datos de cada uno
    int P = ns * (ns - 1) / 2; int *plens = malloc(sizeof(int) * P), *popt = malloc(sizeof(int) * P);
    long hlen[512] = {0}, hmoved[16] = {0}, hthrough = 0, extra = 0; int pi = 0;
    static int pairlen[128][128];
    for (int a = 0; a < ns; a++) for (int b = a + 1; b < ns; b++) {
        int opt; int L = build_path(S[a], S[b], &opt);
        pairlen[a][b] = pairlen[b][a] = L;
        plens[pi] = L; popt[pi] = opt; pi++;
        hlen[L < 511 ? L : 511]++; hmoved[__builtin_popcountll(S[a] & ~S[b])]++;
        if (L > opt) extra++;
        int th = 0;
        for (int k = 0; k <= L; k++) { gget(path[k])->uses++; if (k > 0 && k < L && sol_index(path[k]) >= 0) th = 1; }
        hthrough += th;
    }
    printf("  caminos por pareja (%d parejas):\n    reinas que cambian de sitio (A AND NOT B):", P);
    for (int k = 0; k < 16; k++) if (hmoved[k]) printf("  %d:%ld", k, hmoved[k]);
    printf("\n    longitud del camino (saltos):");
    for (int k = 0; k < 512; k++) if (hlen[k]) printf("  %d:%ld", k, hlen[k]);
    printf("\n    caminos más largos que su emparejamiento ideal (por bloqueos): %ld de %d\n", extra, P);
    printf("    caminos que pasan por otra solución por el camino: %ld de %d\n", hthrough, P);

    // grafo unión de todos los caminos de pareja
    long hu[64] = {0}, intermed = 0; long hconf[64] = {0}, hconf_shared[64] = {0};
    for (uint64_t h = 0; h < gcap; h++) if (G[h].key) {
        int u = G[h].uses; if (sol_index(G[h].key) >= 0) continue;
        intermed++; hu[u < 63 ? u : 63]++;
        int c = conflicts(G[h].key); hconf[c < 63 ? c : 63]++; if (u >= 2) hconf_shared[c < 63 ? c : 63]++;
    }
    printf("  grafo de todos los caminos: %ld tableros intermedios distintos\n    nº de caminos de pareja que pasan por cada intermedio:", intermed);
    for (int k = 1; k < 64; k++) if (hu[k]) printf("  %d:%ld", k, hu[k]);
    printf("\n    pares de reinas atacándose en los intermedios (todos):");
    for (int k = 0; k < 64; k++) if (hconf[k]) printf("  %d:%ld", k, hconf[k]);
    printf("\n    ídem, solo en los compartidos por ≥2 caminos:");
    for (int k = 0; k < 64; k++) if (hconf_shared[k]) printf("  %d:%ld", k, hconf_shared[k]);
    printf("\n");

    // 2) secuencias al azar: de solución en solución sin repetir, hasta pasar por todas
    long *tour = malloc(sizeof(long) * R); long tsum = 0;
    for (int r = 0; r < R; r++) {
        int vis[128] = {0}, cur = rnd() % ns, left = ns - 1; long len = 0; vis[cur] = 1;
        while (left) { int nxt; do nxt = rnd() % ns; while (vis[nxt]); len += pairlen[cur][nxt]; vis[nxt] = 1; cur = nxt; left--; }
        tour[r] = len; tsum += len;
    }
    qsort(tour, R, sizeof(long), cmpl);
    // recorrido mínimo con estos caminos (programación dinámica si cabe)
    long best = -1;
    if (ns <= 20) {
        int Sm = 1 << ns; int *dp = malloc(sizeof(int) * Sm * ns); for (long i = 0; i < (long)Sm * ns; i++) dp[i] = 1 << 28;
        for (int a = 0; a < ns; a++) dp[(1 << a) * ns + a] = 0;
        for (int m = 1; m < Sm; m++) for (int a = 0; a < ns; a++) if ((m >> a & 1) && dp[m * ns + a] < (1 << 28))
            for (int b = 0; b < ns; b++) if (!(m >> b & 1)) { int v = dp[m * ns + a] + pairlen[a][b]; if (v < dp[(m | 1 << b) * ns + b]) dp[(m | 1 << b) * ns + b] = v; }
        best = 1 << 28; for (int a = 0; a < ns; a++) if (dp[(Sm - 1) * ns + a] < best) best = dp[(Sm - 1) * ns + a];
        free(dp);
    }
    printf("  secuencias al azar por todas las soluciones (%d): saltos  mín %ld  mediana %ld  media %.1f  máx %ld", R, tour[0], tour[R / 2], (double)tsum / R, tour[R - 1]);
    if (best >= 0) printf("   (la mejor posible con estos caminos: %ld)", best);
    printf("\n");
    return 0;
}
