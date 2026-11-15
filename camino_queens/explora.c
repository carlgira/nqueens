// explora.c — explorar el grafo de saltos desde una solución hasta encontrarlas todas.
//
// Tablero = N reinas en casillas distintas. Salto = una reina a una casilla
// vecina libre (8 direcciones). Desde una solución inicial se van expandiendo
// los vecinos en orden aleatorio; cada tablero se guarda UNA vez en una tabla
// hash junto con su padre (de dónde vino) y su profundidad, así nunca se
// procesa dos veces y el camino a cualquier tablero se reconstruye siguiendo
// los padres. Se para cuando han aparecido todas las soluciones.
//
//   prof    en profundidad: se expande siempre el último descubierto (pila)
//   anch    en anchura: se expande el más antiguo (cola)
//
// Se repite R veces (solución inicial y orden al azar) y se resume:
//   - tableros descubiertos hasta tener todas las soluciones
//   - en qué orden aparecen las soluciones y a qué profundidad del árbol,
//     comparada con la distancia mínima real (búsqueda en anchura exacta)
//
// Uso: ./explora N modo R [semilla]      (N ≤ 8; modo: prof | anch)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int N, NC;
static uint64_t smask[128]; static int ns, xs[8];
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void bt(int c, int rows, int d1, int d2) {
    if (c == N) { uint64_t m = 0; for (int i = 0; i < N; i++) m |= 1ULL << (xs[i] * N + i); smask[ns++] = m; return; }
    for (int r = 0; r < N; r++) {
        if ((rows >> r & 1) || (d1 >> (r + c) & 1) || (d2 >> (r - c + N) & 1)) continue;
        xs[c] = r; bt(c + 1, rows | 1 << r, d1 | 1 << (r + c), d2 | 1 << (r - c + N));
    }
}

// ---------- tabla hash: tablero -> (padre, profundidad) ----------
typedef struct { uint64_t key; int parent, depth; } Node;     // key 0 = vacío (ningún tablero es 0)
static Node *T; static uint64_t cap; static long nnodes;
static uint64_t h64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static long find_or_add(uint64_t key, int parent, int depth, int *isnew) {
    uint64_t h = h64(key) & (cap - 1);
    while (T[h].key) { if (T[h].key == key) { *isnew = 0; return (long)h; } h = (h + 1) & (cap - 1); }
    T[h].key = key; T[h].parent = parent; T[h].depth = depth; *isnew = 1; return (long)h;
}

static const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
static int neighbors(uint64_t m, uint64_t *out) {
    int k = 0;
    for (int p = 0; p < NC; p++) if (m >> p & 1) {
        int r = p / N, c = p % N;
        for (int d = 0; d < 8; d++) {
            int r2 = r + dr[d], c2 = c + dc[d];
            if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue;
            int q = r2 * N + c2; if (m >> q & 1) continue;
            out[k++] = (m & ~(1ULL << p)) | (1ULL << q);
        }
    }
    return k;
}
static int sol_index(uint64_t m) { for (int k = 0; k < ns; k++) if (smask[k] == m) return k; return -1; }

// distancias exactas entre soluciones (búsqueda en anchura completa desde cada una)
static int D[128][128];
static void exact_distances(void) {
    for (int a = 0; a < ns; a++) {
        memset(T, 0, cap * sizeof(Node)); nnodes = 0;
        uint64_t *q = malloc(sizeof(uint64_t) * (cap / 2 + 1)); long h = 0, t = 0; int isnew;
        find_or_add(smask[a], -1, 0, &isnew); q[t++] = smask[a];
        int found = 1; D[a][a] = 0;
        while (h < t && found < ns) {
            uint64_t m = q[h++]; int isn; long sm = find_or_add(m, 0, 0, &isn); int d = T[sm].depth;
            uint64_t nb[64]; int k = neighbors(m, nb);
            for (int i = 0; i < k; i++) { find_or_add(nb[i], 0, d + 1, &isnew); if (isnew) { q[t++] = nb[i]; int s = sol_index(nb[i]); if (s >= 0) { D[a][s] = d + 1; found++; } } }
        }
        free(q);
    }
}

static int cmp(const void *a, const void *b) { long x = *(const long *)a, y = *(const long *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: ./explora N prof|anch R [semilla]\n"); return 1; }
    N = atoi(argv[1]); NC = N * N; int dfs = !strcmp(argv[2], "prof"); int R = atoi(argv[3]);
    if (argc > 4) rs ^= (uint64_t)atoll(argv[4]) * 0x9E3779B97F4A7C15ULL;
    bt(0, 0, 0, 0);
    // capacidad: nº de tableros C(N², N), con margen
    double tot = 1; for (int i = 0; i < N; i++) tot = tot * (NC - i) / (i + 1);
    cap = 1; while (cap < 2 * tot) cap <<= 1;
    T = malloc(cap * sizeof(Node));
    printf("N = %d   soluciones %d   tableros posibles %.0f   modo %s   repeticiones %d\n", N, ns, tot, dfs ? "profundidad" : "anchura", R);
    exact_distances();
    printf("  distancias exactas entre soluciones (saltos): ");
    { long hist[64] = {0}; for (int a = 0; a < ns; a++) for (int b = a + 1; b < ns; b++) hist[D[a][b]]++; for (int d = 0; d < 64; d++) if (hist[d]) printf(" %d:%ld", d, hist[d]); }
    printf("   (pares)\n");

    // matriz de distancias y capas de la búsqueda en anchura completa desde la solución 0
    printf("  matriz de distancias:\n      "); for (int b2 = 0; b2 < ns && b2 < 40; b2++) printf("%3d", b2); printf("\n");
    for (int a2 = 0; a2 < ns && a2 < 40; a2++) { printf("   %2d ", a2); for (int b2 = 0; b2 < ns && b2 < 40; b2++) printf("%3d", D[a2][b2]); printf("\n"); }
    {
        memset(T, 0, cap * sizeof(Node)); uint64_t *q = malloc(sizeof(uint64_t) * (size_t)(tot + 16)); long h = 0, t = 0; int isnew;
        find_or_add(smask[0], -1, 0, &isnew); q[t++] = smask[0];
        long layer[512] = {0}, solin[512] = {0}; int maxl = 0; layer[0] = 1; solin[0] = 1;
        while (h < t) {
            uint64_t m = q[h++]; int isn; long sm = find_or_add(m, 0, 0, &isn); int d = T[sm].depth;
            uint64_t nb[64]; int k = neighbors(m, nb);
            for (int i = 0; i < k; i++) { find_or_add(nb[i], 0, d + 1, &isnew); if (isnew) { q[t++] = nb[i]; layer[d + 1]++; if (d + 1 > maxl) maxl = d + 1; if (sol_index(nb[i]) >= 0) solin[d + 1]++; } }
        }
        printf("  capas desde la solución 0 (distancia: tableros [soluciones]):");
        for (int d = 0; d <= maxl; d++) { printf(" %d:%ld", d, layer[d]); if (solin[d]) printf("[%ld]", solin[d]); }
        printf("\n  tablero más lejano de una solución: %d saltos\n", maxl);
        free(q);
    }

    uint64_t *stack = malloc(sizeof(uint64_t) * (size_t)(tot + 16));
    int K = ns - 1;                                     // soluciones a encontrar (sin la inicial)
    long *endx = malloc(sizeof(long) * R), *maxdep = malloc(sizeof(long) * R);
    long *discat = malloc(sizeof(long) * R * K), *depat = malloc(sizeof(long) * R * K), *gap = malloc(sizeof(long) * R * K);
    char (*ordr)[512] = malloc((size_t)R * 512), (*patt)[512] = malloc((size_t)R * 512);
    long firstd[64] = {0};
    for (int rep = 0; rep < R; rep++) {
        memset(T, 0, cap * sizeof(Node));
        int start = rnd() % ns; long h = 0, t = 0; int isnew, found = 1;
        find_or_add(smask[start], -1, 0, &isnew); stack[t++] = smask[start];
        long disc = 1, md = 0; char *o = ordr[rep], *pt = patt[rep]; int lo = 0, lp = 0;
        lo += sprintf(o + lo, "%d:", start);
        while (found < ns && h < t) {
            uint64_t m = dfs ? stack[--t] : stack[h++];
            int isn; long sm = find_or_add(m, 0, 0, &isn); int d = T[sm].depth;
            uint64_t nb[64]; int k = neighbors(m, nb);
            for (int i = k - 1; i > 0; i--) { int j = rnd() % (i + 1); uint64_t x = nb[i]; nb[i] = nb[j]; nb[j] = x; }
            for (int i = 0; i < k && found < ns; i++) {
                long sl = find_or_add(nb[i], (int)sm, d + 1, &isnew);
                if (!isnew) continue;
                disc++; stack[t++] = nb[i]; if (d + 1 > md) md = d + 1;
                int s = sol_index(nb[i]);
                if (s >= 0) {
                    int kk = found - 1;
                    discat[rep * K + kk] = disc; depat[rep * K + kk] = T[sl].depth;
                    gap[rep * K + kk] = disc - (kk ? discat[rep * K + kk - 1] : 1);
                    if (found == 1) firstd[D[start][s]]++;
                    if (lo < 480) lo += sprintf(o + lo, " %d", s);
                    if (lp < 480) lp += sprintf(pt + lp, "%s%d", lp ? "," : "", D[start][s]);
                    found++;
                }
            }
        }
        endx[rep] = disc; maxdep[rep] = md;
    }
    // ---------- resumen ----------
    #define QS(arr, n) do { qsort(arr, n, sizeof(long), cmp); } while (0)
    long *tmp = malloc(sizeof(long) * R);
    #define PQ(lab, get) do { for (int r_ = 0; r_ < R; r_++) tmp[r_] = (get); QS(tmp, R); \
        printf("  %-46s mín %8ld   10%% %8ld   mediana %8ld   90%% %8ld   máx %8ld\n", lab, tmp[0], tmp[R / 10], tmp[R / 2], tmp[9 * R / 10], tmp[R - 1]); } while (0)
    printf("\n  --- %d experimentos al azar ---\n", R);
    PQ("tableros explorados hasta tener todas", endx[r_]);
        PQ("profundidad máxima del árbol al terminar", maxdep[r_]);
    for (int k = 0; k < K; k++) { char lab[64]; snprintf(lab, sizeof lab, "explorados al aparecer la solución n.º %d", k + 1); PQ(lab, discat[r_ * K + k]); }
    for (int k = 0; k < K; k++) { char lab[64]; snprintf(lab, sizeof lab, "profundidad (camino) de la solución n.º %d", k + 1); PQ(lab, depat[r_ * K + k]); }
    for (int k = 1; k < K; k++) { char lab[64]; snprintf(lab, sizeof lab, "tableros entre la %d.ª y la %d.ª", k, k + 1); PQ(lab, gap[r_ * K + k]); }
    printf("  primera solución encontrada según su distancia al inicio:");
    for (int d = 0; d < 64; d++) if (firstd[d]) printf("  d=%d: %.1f %%", d, 100.0 * firstd[d] / R); printf("\n");
    // órdenes
    for (int which = 0; which < 2; which++) {
        char (*A)[512] = which ? patt : ordr;
        int *idx = malloc(sizeof(int) * R); for (int r = 0; r < R; r++) idx[r] = r;
        // ordenar por cadena para contar
        for (int i = 1; i < R; i++) { int v = idx[i], j = i - 1; while (j >= 0 && strcmp(A[idx[j]], A[v]) > 0) { idx[j + 1] = idx[j]; j--; } idx[j + 1] = v; }
        int nd = 0; int (*cnt)[2] = malloc(sizeof(int) * 2 * R);
        for (int i = 0; i < R;) { int j = i; while (j < R && !strcmp(A[idx[j]], A[idx[i]])) j++; cnt[nd][0] = idx[i]; cnt[nd][1] = j - i; nd++; i = j; }
        for (int i = 1; i < nd; i++) { int v0 = cnt[i][0], v1 = cnt[i][1], j = i - 1; while (j >= 0 && cnt[j][1] < v1) { cnt[j + 1][0] = cnt[j][0]; cnt[j + 1][1] = cnt[j][1]; j--; } cnt[j + 1][0] = v0; cnt[j + 1][1] = v1; }
        printf("  %s: %d distintos en %d experimentos; los más frecuentes:\n", which ? "secuencias de distancias al inicio" : "órdenes exactos (inicio: soluciones en orden)", nd, R);
        for (int i = 0; i < nd && i < 6; i++) printf("     %4d×  %s\n", cnt[i][1], A[cnt[i][0]]);
        free(idx); free(cnt);
    }
    return 0;
}
