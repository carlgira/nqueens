// salto_rey.c — caminos de decisión entre soluciones moviendo UNA reina por paso.
//
// Tablero = N reinas en casillas distintas de N×N (no hace falta una por columna).
// Un salto = mover una reina a una de sus 8 casillas vecinas, si está libre.
// Búsqueda en anchura exacta: distancias entre todas las soluciones, un camino
// más corto para cada par, y el recorrido más corto que pasa por todas.
//
// Uso: ./salto_rey N        (N ≤ 5: el tablero cabe en una máscara de 25 bits)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N, NC;
static uint32_t smask[64]; static int ns;
static int xs[8];

static void bt(int c, int rows, int d1, int d2) {
    if (c == N) { uint32_t m = 0; for (int i = 0; i < N; i++) m |= 1u << (xs[i] * N + i); smask[ns++] = m; return; }
    for (int r = 0; r < N; r++) {
        if ((rows >> r & 1) || (d1 >> (r + c) & 1) || (d2 >> (r - c + N) & 1)) continue;
        xs[c] = r; bt(c + 1, rows | 1 << r, d1 | 1 << (r + c), d2 | 1 << (r - c + N));
    }
}
static int sol_index(uint32_t m) { for (int k = 0; k < ns; k++) if (smask[k] == m) return k; return -1; }
static void show(uint32_t m) {                      // fila de cada columna, o el tablero
    printf("[");
    for (int r = 0; r < N; r++) { for (int c = 0; c < N; c++) printf("%c", m >> (r * N + c) & 1 ? 'Q' : '.'); if (r + 1 < N) printf("|"); }
    printf("]");
}

static unsigned char *dist;                          // 2^(N²) bytes
static uint32_t *queue;
static void bfs(uint32_t src) {
    memset(dist, 255, (size_t)1 << NC);
    long h = 0, t = 0; queue[t++] = src; dist[src] = 0;
    static const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    while (h < t) {
        uint32_t m = queue[h++]; int d = dist[m];
        for (int p = 0; p < NC; p++) if (m >> p & 1) {
            int r = p / N, c = p % N;
            for (int k = 0; k < 8; k++) {
                int r2 = r + dr[k], c2 = c + dc[k];
                if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue;
                int q = r2 * N + c2; if (m >> q & 1) continue;
                uint32_t m2 = (m & ~(1u << p)) | (1u << q);
                if (dist[m2] == 255) { dist[m2] = d + 1; queue[t++] = m2; }
            }
        }
    }
}
// un camino más corto de a hasta b, usando dist calculado desde b; cuenta caminos más cortos
static const char *DIRN[8] = {"↖", "↑", "↗", "←", "→", "↙", "↓", "↘"};
static void path(uint32_t a, int verbose) {
    static const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    uint32_t m = a; int step = 0;
    while (dist[m]) {
        int moved = 0;
        for (int p = 0; p < NC && !moved; p++) if (m >> p & 1) {
            int r = p / N, c = p % N;
            for (int k = 0; k < 8 && !moved; k++) {
                int r2 = r + dr[k], c2 = c + dc[k];
                if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue;
                int q = r2 * N + c2; if (m >> q & 1) continue;
                uint32_t m2 = (m & ~(1u << p)) | (1u << q);
                if (dist[m2] == dist[m] - 1) {
                    step++;
                    if (verbose) { printf("      %2d. reina (%d,%d) %s (%d,%d)", step, r, c, DIRN[k], r2, c2); int s = sol_index(m2); if (s >= 0) printf("   ← solución %d", s); printf("\n"); }
                    m = m2; moved = 1;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    N = argc > 1 ? atoi(argv[1]) : 5; NC = N * N;
    if (NC > 25) { fprintf(stderr, "N ≤ 5\n"); return 1; }
    bt(0, 0, 0, 0);
    dist = malloc((size_t)1 << NC); queue = malloc(sizeof(uint32_t) * 4000000UL);
    printf("N = %d   soluciones %d\n", N, ns);
    for (int k = 0; k < ns; k++) { printf("  %d: filas por columna", k); for (int c = 0; c < N; c++) for (int r = 0; r < N; r++) if (smask[k] >> (r * N + c) & 1) printf(" %d", r); printf("   "); show(smask[k]); printf("\n"); }
    int D[64][64]; long reach = 0;
    for (int a = 0; a < ns; a++) {
        bfs(smask[a]);
        if (a == 0) { for (long m = 0; m < (1L << NC); m++) reach += dist[m] != 255; }
        for (int b = 0; b < ns; b++) D[a][b] = dist[smask[b]];
    }
    printf("  tableros alcanzables desde una solución: %ld\n", reach);
    printf("  distancias (nº de saltos) entre soluciones:\n      ");
    for (int b = 0; b < ns; b++) printf("%3d", b); printf("\n");
    for (int a = 0; a < ns; a++) { printf("   %2d ", a); for (int b = 0; b < ns; b++) printf("%3d", D[a][b]); printf("\n"); }
    // recorrido más corto por todas (camino abierto), programación dinámica
    int S = 1 << ns; int *dp = malloc(sizeof(int) * S * ns), *par = malloc(sizeof(int) * S * ns);
    for (int i = 0; i < S * ns; i++) dp[i] = 1 << 28;
    for (int a = 0; a < ns; a++) dp[(1 << a) * ns + a] = 0;
    for (int m = 1; m < S; m++) for (int a = 0; a < ns; a++) if ((m >> a & 1) && dp[m * ns + a] < (1 << 28))
        for (int b = 0; b < ns; b++) if (!(m >> b & 1)) {
            int m2 = m | 1 << b, v = dp[m * ns + a] + D[a][b];
            if (v < dp[m2 * ns + b]) { dp[m2 * ns + b] = v; par[m2 * ns + b] = a; }
        }
    int best = 1 << 28, end = 0; for (int a = 0; a < ns; a++) if (dp[(S - 1) * ns + a] < best) { best = dp[(S - 1) * ns + a]; end = a; }
    int order[64], m = S - 1, cur = end; for (int k = ns - 1; k >= 0; k--) { order[k] = cur; int p = par[m * ns + cur]; m &= ~(1 << cur); cur = p; }
    printf("  recorrido más corto por las %d soluciones: %d saltos\n    orden:", ns, best);
    for (int k = 0; k < ns; k++) printf(" %d", order[k]); printf("\n    saltos de cada tramo:");
    for (int k = 0; k + 1 < ns; k++) printf(" %d", D[order[k]][order[k + 1]]); printf("\n");
    printf("  movimientos de cada tramo (uno de los caminos más cortos):\n");
    for (int k = 0; k + 1 < ns; k++) {
        printf("    tramo %d → %d (%d saltos)\n", order[k], order[k + 1], D[order[k]][order[k + 1]]);
        bfs(smask[order[k + 1]]); path(smask[order[k]], 1);
    }
    return 0;
}
