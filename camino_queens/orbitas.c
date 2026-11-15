// orbitas.c — reducir el problema a un representante por grupo de simetría (D4).
//
// Las 8 simetrías del tablero conservan los saltos de rey, así que la
// distancia entre dos grupos es la mínima entre el representante de uno y las
// 8 versiones del otro. Se mide:
//   1) caminos por bits (como salto_bits.c) entre grupos, y el recorrido más
//      corto que pasa por un miembro de cada grupo (programación dinámica con
//      la orientación del miembro como parte del estado);
//   2) exploración con memoria (como explora.c) parando cuando ha aparecido una
//      solución de cada grupo, frente a parar cuando han aparecido todas.
//
// Uso: ./orbitas N [repeticiones_exploración]     (N ≤ 8; la exploración solo N ≤ 6)

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


// ---------- simetrías sobre máscaras ----------
static uint64_t d4m(uint64_t m, int g) {
    uint64_t o = 0;
    for (int p = 0; p < NC; p++) if (m >> p & 1) {
        int r = p / N, c = p % N;
        if (g & 1) { int t = r; r = c; c = t; }
        if (g & 2) c = N - 1 - c;
        if (g & 4) r = N - 1 - r;
        o |= 1ULL << (r * N + c);
    }
    return o;
}
static int orb[128], rep[64], osz[64], no;

// ---------- exploración con memoria ----------
typedef struct { uint64_t key; int depth; } XN;
static XN *X; static uint64_t xcap;
static long xfind(uint64_t k, int depth, int *isnew) {
    uint64_t h = h64(k) & (xcap - 1);
    while (X[h].key && X[h].key != k) h = (h + 1) & (xcap - 1);
    if (!X[h].key) { X[h].key = k; X[h].depth = depth; *isnew = 1; } else *isnew = 0;
    return (long)h;
}
static int neighbors(uint64_t m, uint64_t *out) {
    static const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int k = 0;
    for (int p = 0; p < NC; p++) if (m >> p & 1) {
        int r = p / N, c = p % N;
        for (int d = 0; d < 8; d++) { int r2 = r + dr[d], c2 = c + dc[d]; if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue;
            int q = r2 * N + c2; if (m >> q & 1) continue; out[k++] = (m & ~(1ULL << p)) | (1ULL << q); }
    }
    return k;
}
// devuelve tableros descubiertos hasta tener (por_grupo ? un miembro de cada grupo : todas)
static long explore(int dfs, int por_grupo, uint64_t *stack) {
    memset(X, 0, xcap * sizeof(XN));
    int start = rnd() % ns; int got[128] = {0}, gotg[64] = {0}, need = por_grupo ? no : ns, have = 1;
    got[start] = 1; gotg[orb[start]] = 1;
    long h = 0, t = 0, disc = 1; int isnew; xfind(S[start], 0, &isnew); stack[t++] = S[start];
    while (have < need && h < t) {
        uint64_t m = dfs ? stack[--t] : stack[h++]; long sm = xfind(m, 0, &isnew); int d = X[sm].depth;
        uint64_t nb[64]; int k = neighbors(m, nb);
        for (int i = k - 1; i > 0; i--) { int j = rnd() % (i + 1); uint64_t x = nb[i]; nb[i] = nb[j]; nb[j] = x; }
        for (int i = 0; i < k && have < need; i++) {
            xfind(nb[i], d + 1, &isnew); if (!isnew) continue;
            disc++; stack[t++] = nb[i];
            int s = sol_index(nb[i]); if (s < 0) continue;
            if (por_grupo) { if (!gotg[orb[s]]) { gotg[orb[s]] = 1; have++; } }
            else if (!got[s]) { got[s] = 1; have++; }
        }
    }
    return disc;
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); NC = N * N; int R = argc > 2 ? atoi(argv[2]) : 0;
    bt(0, 0, 0, 0);
    // grupos de simetría
    no = 0; for (int s = 0; s < ns; s++) orb[s] = -1;
    for (int s = 0; s < ns; s++) if (orb[s] < 0) {
        rep[no] = s; osz[no] = 0;
        for (int g = 0; g < 8; g++) { int t = sol_index(d4m(S[s], g)); if (t >= 0 && orb[t] < 0) { orb[t] = no; osz[no]++; } }
        no++;
    }
    // canónico: el miembro de máscara mínima; grupos ordenados por ese número
    for (int o = 0; o < no; o++) { uint64_t mn = ~0ULL; for (int g = 0; g < 8; g++) { uint64_t v = d4m(S[rep[o]], g); if (v < mn) mn = v; } rep[o] = sol_index(mn); }
    for (int a = 1; a < no; a++) { int ra = rep[a], sa = osz[a], b = a - 1; while (b >= 0 && S[rep[b]] > S[ra]) { rep[b + 1] = rep[b]; osz[b + 1] = osz[b]; b--; } rep[b + 1] = ra; osz[b + 1] = sa; }
    for (int s = 0; s < ns; s++) for (int o = 0; o < no; o++) for (int g = 0; g < 8; g++) if (d4m(S[rep[o]], g) == S[s]) orb[s] = o;
    printf("N = %d   soluciones %d   grupos de simetría %d (tamaños:", N, ns, no);
    for (int o = 0; o < no; o++) printf(" %d", osz[o]); printf(")\n");

    // 1) caminos por bits entre grupos
    static int C[24][8][24][8];          // coste desde (grupo X, orientación g) a (grupo Y, orientación h)
    if (no > 24) { fprintf(stderr, "demasiados grupos\n"); return 1; }
    gcap = 1 << 20; G = calloc(gcap, sizeof(Node));
    long hlen[64] = {0}; int npairs = 0;
    for (int x = 0; x < no; x++) for (int g = 0; g < 8; g++) for (int y = 0; y < no; y++) for (int h = 0; h < 8; h++) {
        int opt; C[x][g][y][h] = build_path(d4m(S[rep[x]], g), d4m(S[rep[y]], h), &opt);
    }
    printf("  distancia entre grupos (mínima entre orientaciones) y dentro de cada grupo:\n      ");
    for (int y = 0; y < no; y++) printf("%3d", y); printf("\n");
    for (int x = 0; x < no; x++) {
        printf("   %2d ", x);
        for (int y = 0; y < no; y++) {
            int best = 1 << 28, bh = 0;
            for (int h = 0; h < 8; h++) { if (x == y && d4m(S[rep[y]], h) == S[rep[x]]) continue; if (C[x][0][y][h] < best) { best = C[x][0][y][h]; bh = h; } }
            if (best == 1 << 28) { printf("  -"); continue; }
            printf("%3d", best);
            if (x < y) { int opt; int L = build_path(S[rep[x]], d4m(S[rep[y]], bh), &opt); for (int k = 1; k < L; k++) gget(path[k])->uses++; hlen[L]++; npairs++; }
        }
        printf("\n");
    }
    long inter = 0; for (uint64_t h = 0; h < gcap; h++) if (G[h].key && sol_index(G[h].key) < 0) inter++;
    printf("  parejas de grupos %d (frente a %d parejas de soluciones); tableros intermedios usados: %ld\n", npairs, ns * (ns - 1) / 2, inter);
    // recorrido más corto por un miembro de cada grupo
    if (no >= 2 && no <= 14) {
        int Sm = 1 << no; size_t sz = (size_t)Sm * no * 8; int *dp = malloc(sizeof(int) * sz);
        for (size_t i = 0; i < sz; i++) dp[i] = 1 << 28;
        for (int x = 0; x < no; x++) dp[((size_t)(1 << x) * no + x) * 8 + 0] = 0;     // la orientación inicial se puede fijar
        for (int m = 1; m < Sm; m++) for (int x = 0; x < no; x++) if (m >> x & 1) for (int g = 0; g < 8; g++) {
            int v = dp[((size_t)m * no + x) * 8 + g]; if (v >= 1 << 28) continue;
            for (int y = 0; y < no; y++) if (!(m >> y & 1)) for (int h = 0; h < 8; h++) {
                int w = v + C[x][g][y][h]; size_t id = ((size_t)(m | 1 << y) * no + y) * 8 + h; if (w < dp[id]) dp[id] = w; }
        }
        int best = 1 << 28, ex = 0, eg = 0; for (int x = 0; x < no; x++) for (int g = 0; g < 8; g++) { int v = dp[((size_t)(Sm - 1) * no + x) * 8 + g]; if (v < best) { best = v; ex = x; eg = g; } }
        printf("  recorrido más corto por un miembro de cada grupo: %d saltos (%d tramos)\n", best, no - 1);
        // reconstruir uno hacia atrás
        { int ordx[24], jumps[24], m = Sm - 1, x = ex, g = eg, k = no - 1; ordx[k] = x;
          while (k > 0) { int v = dp[((size_t)m * no + x) * 8 + g], found = 0; int m0 = m & ~(1 << x);
              for (int px = 0; px < no && !found; px++) if (m0 >> px & 1) for (int pg = 0; pg < 8 && !found; pg++) {
                  int pv = dp[((size_t)m0 * no + px) * 8 + pg]; if (pv < (1 << 28) && pv + C[px][pg][x][g] == v) { jumps[k] = C[px][pg][x][g]; x = px; g = pg; m = m0; found = 1; } }
              ordx[--k] = x; }
          printf("     orden:"); for (int i = 0; i < no; i++) printf(" %d", ordx[i]);
          printf("   saltos:"); for (int i = 1; i < no; i++) printf(" %d", jumps[i]); printf("\n"); }
        // cuántos órdenes de grupos distintos alcanzan el óptimo (contados sin orientación)
        if (no <= 10) {
            int perm2[24]; for (int i = 0; i < no; i++) perm2[i] = i; long nopt = 0, ntot = 0;
            // recorrer permutaciones con orientaciones óptimas por programación dinámica en cadena
            for (;;) {
                int cur[8]; for (int h = 0; h < 8; h++) cur[h] = h == 0 ? 0 : 1 << 28;
                for (int i = 1; i < no; i++) { int nx[8]; for (int h = 0; h < 8; h++) { nx[h] = 1 << 28; for (int g = 0; g < 8; g++) if (cur[g] < 1 << 28) { int v = cur[g] + C[perm2[i - 1]][g][perm2[i]][h]; if (v < nx[h]) nx[h] = v; } } memcpy(cur, nx, sizeof cur); }
                int mn = 1 << 28; for (int h = 0; h < 8; h++) if (cur[h] < mn) mn = cur[h];
                ntot++; if (mn == best) nopt++;
                int i = no - 2; while (i >= 0 && perm2[i] > perm2[i + 1]) i--; if (i < 0) break;
                int j = no - 1; while (perm2[j] < perm2[i]) j--; int t = perm2[i]; perm2[i] = perm2[j]; perm2[j] = t;
                for (int a = i + 1, b = no - 1; a < b; a++, b--) { t = perm2[a]; perm2[a] = perm2[b]; perm2[b] = t; }
            }
            printf("     órdenes de grupos que alcanzan el óptimo: %ld de %ld (cada uno cuenta también al revés)\n", nopt, ntot);
        }
        free(dp);
    }
    // secuencias al azar por grupos, eligiendo en cada tramo la orientación más cercana
    { int RR = 20000; long *v = malloc(sizeof(long) * RR);
      for (int r = 0; r < RR; r++) { int vis[64] = {0}, x = rnd() % no, g = 0, left = no - 1; long L = 0; vis[x] = 1;
          while (left) { int y; do y = rnd() % no; while (vis[y]); int bh = 0; for (int h = 1; h < 8; h++) if (C[x][g][y][h] < C[x][g][y][bh]) bh = h; L += C[x][g][y][bh]; x = y; g = bh; vis[y] = 1; left--; }
          v[r] = L; }
      qsort(v, RR, sizeof(long), cmpl);
      printf("  secuencias al azar por los grupos: saltos  mín %ld  mediana %ld  máx %ld\n", v[0], v[RR / 2], v[RR - 1]); free(v); }

    // ---------- trayectorias reproducibles entre grupos ----------
    if (no >= 2) {
        printf("  representantes canónicos (filas por columna):\n");
        for (int o = 0; o < no; o++) { printf("     grupo %2d (%d):", o, osz[o]); for (int c = 0; c < N; c++) for (int r = 0; r < N; r++) if (S[rep[o]] >> (r * N + c) & 1) printf(" %d", r); printf("\n"); }
        int Dg[24][24], Hg[24][24], Kg[24][24];
        for (int x = 0; x < no; x++) for (int y = 0; y < no; y++) {
            int best = 1 << 28, bh = 0; for (int h = 0; h < 8; h++) if (C[x][0][y][h] < best) { best = C[x][0][y][h]; bh = h; }
            Dg[x][y] = best; Hg[x][y] = bh; Kg[x][y] = __builtin_popcountll(S[rep[x]] & ~d4m(S[rep[y]], bh));
        }
        // salto frente a reinas movidas
        long tab[16][32]; memset(tab, 0, sizeof tab);
        for (int x = 0; x < no; x++) for (int y = x + 1; y < no; y++) tab[Kg[x][y]][Dg[x][y] < 31 ? Dg[x][y] : 31]++;
        printf("  salto entre grupos según cuántas reinas cambian de sitio (reinas: saltos×parejas):\n");
        for (int k = 0; k < 16; k++) { int any = 0; for (int d = 0; d < 32; d++) if (tab[k][d]) any = 1; if (!any) continue;
            printf("     %d reinas:", k); for (int d = 0; d < 32; d++) if (tab[k][d]) printf("  %d×%ld", d, tab[k][d]); printf("\n"); }
        // árbol mínimo (Prim desde el grupo 0)
        int in[24] = {0}, par[24], key[24], deg[24] = {0}; for (int i = 0; i < no; i++) { key[i] = 1 << 28; par[i] = -1; } key[0] = 0;
        int tsum = 0; long lh[32] = {0};
        for (int it = 0; it < no; it++) {
            int u = -1; for (int i = 0; i < no; i++) if (!in[i] && (u < 0 || key[i] < key[u])) u = i;
            in[u] = 1; if (par[u] >= 0) { tsum += key[u]; deg[u]++; deg[par[u]]++; lh[key[u]]++; }
            for (int v = 0; v < no; v++) if (!in[v] && Dg[u][v] < key[v]) { key[v] = Dg[u][v]; par[v] = u; }
        }
        printf("  árbol mínimo entre grupos: %d saltos en total; aristas", tsum);
        for (int v = 1; v < no; v++) printf(" %d-%d(%d)", par[v], v, Dg[par[v]][v]);
        printf("\n     tamaños de salto en el árbol:"); for (int d = 0; d < 32; d++) if (lh[d]) printf(" %d×%ld", d, lh[d]);
        printf("   grados:"); for (int i = 0; i < no; i++) printf(" %d", deg[i]);
        // diámetro del árbol (en aristas) por BFS doble
        { int adj[24][24] = {{0}}; for (int v = 1; v < no; v++) adj[v][par[v]] = adj[par[v]][v] = 1;
          int far = 0, dmax = 0; for (int pass = 0; pass < 2; pass++) { int dist[24]; for (int i = 0; i < no; i++) dist[i] = -1; int q[24], h = 0, t = 0; q[t++] = far; dist[far] = 0;
              while (h < t) { int u = q[h++]; for (int v = 0; v < no; v++) if (adj[u][v] && dist[v] < 0) { dist[v] = dist[u] + 1; q[t++] = v; } }
              for (int i = 0; i < no; i++) if (dist[i] > dmax || (pass == 0 && dist[i] > dist[far])) { far = i; dmax = dist[i]; } }
          printf("   diámetro: %d aristas\n", dmax); }
        // regla del más cercano, con orientación
        { int vis[24] = {0}, x = 0, g = 0, L = 0; vis[0] = 1; printf("  regla 'al grupo más cercano': 0");
          for (int st = 1; st < no; st++) { int by = -1, bh = 0, bc = 1 << 28;
              for (int y = 0; y < no; y++) if (!vis[y]) for (int h = 0; h < 8; h++) if (C[x][g][y][h] < bc) { bc = C[x][g][y][h]; by = y; bh = h; }
              printf(" -(%d)- %d", bc, by); L += bc; vis[by] = 1; x = by; g = bh; }
          printf("   total %d\n", L); }
    }

    // 2) exploración: parar con un miembro por grupo frente a con todas
    if (R > 0 && N <= 6) {
        double tb = 1; for (int i = 0; i < N; i++) tb = tb * (NC - i) / (i + 1);
        xcap = 1; while (xcap < 2 * tb) xcap <<= 1; X = malloc(xcap * sizeof(XN));
        uint64_t *stack = malloc(sizeof(uint64_t) * (size_t)(tb + 16));
        for (int dfs = 1; dfs >= 0; dfs--) for (int pg = 0; pg < 2; pg++) {
            long *v = malloc(sizeof(long) * R); for (int r = 0; r < R; r++) v[r] = explore(dfs, pg, stack);
            qsort(v, R, sizeof(long), cmpl);
            printf("  exploración en %s, parando con %s: tableros  mín %ld  mediana %ld (%.1f %%)  máx %ld\n", dfs ? "profundidad" : "anchura",
                   pg ? "un miembro de cada grupo" : "todas las soluciones", v[0], v[R / 2], 100.0 * v[R / 2] / tb, v[R - 1]);
            free(v);
        }
    }
    return 0;
}
