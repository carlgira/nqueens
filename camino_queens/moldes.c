// moldes.c — ¿hay una "dirección" predecible para pasar de una solución a la siguiente?
//
// 1) Molde desplazado. Entre dos soluciones s, t el cambio es un molde:
//    quitar = s AND NOT t, poner = t AND NOT s (s XOR t). Repetir el mismo molde
//    devuelve a s, así que se repite DESPLAZADO un vector fijo v cada vez:
//        s_{j+1} = s_j - (quitar + j·v) + (poner + j·v)
//    con o sin dar la vuelta al tablero. Se cuenta cuántas soluciones seguidas
//    encadena, para todas las parejas con pocas reinas cambiadas y todos los v
//    pequeños. Una cadena de 3 o más sería una dirección predecible.
// 2) Saltos iguales. Para cada tamaño de salto d (camino por bits, como en
//    salto_bits.c), grafo de soluciones a distancia exactamente d; cadena más
//    larga sin repetir, frente a un grafo al azar con los mismos grados.
//    Dentro de esas cadenas se mira si el molde se repite (desplazado).
//
// Falso: permutaciones al azar cerradas por D4, tantas como soluciones (con
// las mismas reglas no son soluciones; se usa "pertenece al conjunto").
// Cada cadena se marca "explicada" si algún par está relacionado por D4.
//
// Uso: ./moldes N [k_max] [bases]     (N ≤ 16; bases: 0 = todas, 'w2' en N=14 = casi toroidales W=2)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAXN 16
typedef struct { uint64_t w[4]; } B;
static int N, NC;
static inline int gb(const B *b, int p) { return b->w[p >> 6] >> (p & 63) & 1; }
static inline void sb(B *b, int p) { b->w[p >> 6] |= 1ULL << (p & 63); }
static inline void cb(B *b, int p) { b->w[p >> 6] &= ~(1ULL << (p & 63)); }
static uint64_t hb(const B *b) { uint64_t h = 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 4; i++) { h ^= b->w[i] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); } h ^= h >> 31; h *= 0xff51afd7ed558ccdULL; return h ^ (h >> 29); }
static int beq(const B *a, const B *b) { return !memcmp(a, b, sizeof(B)); }
static int iabs(int a) { return a < 0 ? -a : a; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

// ---------- conjunto de tableros con hash ----------
typedef struct { int n, cap, hcap; B *v; int *slot; } Set;
static void set_init(Set *S, int cap) { S->n = 0; S->cap = cap; S->v = malloc(sizeof(B) * cap); S->hcap = 1; while (S->hcap < 2 * cap) S->hcap <<= 1; S->slot = malloc(sizeof(int) * S->hcap); memset(S->slot, -1, sizeof(int) * S->hcap); }
static int set_find(const Set *S, const B *b) { uint64_t h = hb(b) & (S->hcap - 1); while (S->slot[h] >= 0) { if (beq(&S->v[S->slot[h]], b)) return S->slot[h]; h = (h + 1) & (S->hcap - 1); } return -1; }
static int set_add(Set *S, const B *b) {
    uint64_t h = hb(b) & (S->hcap - 1); while (S->slot[h] >= 0) { if (beq(&S->v[S->slot[h]], b)) return S->slot[h]; h = (h + 1) & (S->hcap - 1); }
    if (S->n == S->cap) { fprintf(stderr, "conjunto lleno\n"); exit(1); }
    S->v[S->n] = *b; S->slot[h] = S->n; return S->n++;
}

// ---------- soluciones ----------
static Set SOL; static int xs[MAXN];
static void bt(int c, unsigned rows, unsigned d1, unsigned d2) {
    if (c == N) { B b = {{0}}; for (int i = 0; i < N; i++) sb(&b, xs[i] * N + i); set_add(&SOL, &b); return; }
    for (int r = 0; r < N; r++) {
        unsigned a = 1u << r, x = 1u << (r + c), e = 1u << (r - c + N);
        if ((rows & a) || (d1 & x) || (d2 & e)) continue;
        xs[c] = r; bt(c + 1, rows | a, d1 | x, d2 | e);
    }
}
static void d4b(const B *m, int g, B *o) {
    memset(o, 0, sizeof(B));
    for (int p = 0; p < NC; p++) if (gb(m, p)) { int r = p / N, c = p % N; if (g & 1) { int t = r; r = c; c = t; } if (g & 2) c = N - 1 - c; if (g & 4) r = N - 1 - r; sb(o, r * N + c); }
}
static int related(const B *a, const B *b) { B y; for (int g = 0; g < 8; g++) { d4b(a, g, &y); if (beq(&y, b)) return 1; } return 0; }
static int seam_pairs(const B *m) {     // W: pares que se atacan en el toro y no en el tablero
    int x[MAXN]; for (int p = 0; p < NC; p++) if (gb(m, p)) x[p % N] = p / N;
    int W = 0; for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) if (iabs(x[i] - x[j]) == N - (j - i)) W++;
    return W;
}

// ---------- 1) moldes desplazados ----------
#define MAXK 16          // reinas cambiadas en los moldes
#define MAXKD 6          // límite para la distancia por bits (emparejamiento por fuerza bruta)
typedef struct { long hist[2][64]; long expl[2], unexpl[2]; int best[2]; B ex[2][8]; int exn[2], exv[2][2], exk[2]; } MStat;
static void mold_chains(const Set *S, int sb_, int tb_, int wrap, MStat *st) {
    const B *s = &S->v[sb_], *t = &S->v[tb_];
    int rem[MAXK], add[MAXK], k = 0, ka = 0;
    for (int p = 0; p < NC; p++) { int a = gb(s, p), b = gb(t, p); if (a && !b) { if (k >= MAXK) return; rem[k++] = p; } if (b && !a) { if (ka >= MAXK) return; add[ka++] = p; } }
    if (k == 0 || k != ka) return;
    for (int dr = -2; dr <= 2; dr++) for (int dc = -2; dc <= 2; dc++) {
        if (!dr && !dc) continue;
        B cur = *t; int idx[64], L = 2; idx[0] = sb_; idx[1] = tb_; B ch[64]; ch[0] = *s; ch[1] = *t;
        for (int j = 1; L < 63; j++) {
            B nx = cur; int ok = 1;
            for (int i = 0; i < k && ok; i++) {
                int r = rem[i] / N + j * dr, c = rem[i] % N + j * dc;
                if (wrap) { r = ((r % N) + N) % N; c = ((c % N) + N) % N; } else if (r < 0 || r >= N || c < 0 || c >= N) { ok = 0; break; }
                int p = r * N + c; if (!gb(&nx, p)) { ok = 0; break; } cb(&nx, p);
            }
            for (int i = 0; i < k && ok; i++) {
                int r = add[i] / N + j * dr, c = add[i] % N + j * dc;
                if (wrap) { r = ((r % N) + N) % N; c = ((c % N) + N) % N; } else if (r < 0 || r >= N || c < 0 || c >= N) { ok = 0; break; }
                int p = r * N + c; if (gb(&nx, p)) { ok = 0; break; } sb(&nx, p);
            }
            if (!ok) break;
            int r = set_find(S, &nx); if (r < 0) break;
            int rep = 0; for (int a = 0; a < L; a++) rep |= idx[a] == r; if (rep) break;
            idx[L] = r; ch[L] = nx; L++; cur = nx;
        }
        if (L >= 3) {
            st->hist[wrap][L]++;
            int e = 0; for (int a = 0; a < L && !e; a++) for (int b = a + 1; b < L && !e; b++) e = related(&ch[a], &ch[b]);
            if (e) st->expl[wrap]++; else st->unexpl[wrap]++;
            if (L > st->best[wrap] && !e) { st->best[wrap] = L; st->exn[wrap] = L < 8 ? L : 8; for (int a = 0; a < st->exn[wrap]; a++) st->ex[wrap][a] = ch[a]; st->exv[wrap][0] = dr; st->exv[wrap][1] = dc; st->exk[wrap] = k; }
        }
    }
}
static void print_board(const B *b) { printf("      "); for (int c = 0; c < N; c++) for (int r = 0; r < N; r++) if (gb(b, r * N + c)) printf(" %d", r); printf("\n"); }

// ---------- 2) distancia por bits (emparejamiento de menor distancia de rey) ----------
static int fr[MAXKD], to_[MAXKD], nk, bestc;
static void asg(int i, int used, int cost) {
    if (cost >= bestc) return; if (i == nk) { bestc = cost; return; }
    for (int j = 0; j < nk; j++) if (!(used >> j & 1)) { int d = iabs(fr[i] / N - to_[j] / N), e = iabs(fr[i] % N - to_[j] % N); asg(i + 1, used | 1 << j, cost + (d > e ? d : e)); }
}
static int bitdist(const B *a, const B *b) {     // -1 si cambian más de MAXK reinas
    nk = 0; int m = 0;
    for (int p = 0; p < NC; p++) { int x = gb(a, p), y = gb(b, p); if (x && !y) { if (nk >= MAXKD) return -1; fr[nk++] = p; } if (y && !x) to_[m++] = p; }
    bestc = 1 << 28; asg(0, 0, 0); return bestc;
}
// cadena más larga sin repetir en un grafo (búsqueda en profundidad con presupuesto)
static int *adj, *adjn, AD; static long budget; static int bestpath, *onpath, *pathv, bestpv[256];
static void lp(int u, int len) {
    if (--budget < 0) return;
    if (len > bestpath) { bestpath = len; memcpy(bestpv, pathv, sizeof(int) * (len < 256 ? len : 256)); }
    for (int i = 0; i < adjn[u]; i++) { int v = adj[u * AD + i]; if (onpath[v]) continue; onpath[v] = 1; if (len < 256) pathv[len] = v; lp(v, len + 1); onpath[v] = 0; if (budget < 0) return; }
}
static int longest(int n, long bud) {
    bestpath = 0; budget = bud;
    // empezar por los de menor grado (extremos probables) y luego por el resto
    int *ord = malloc(sizeof(int) * n); for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = 1; i < n; i++) { int v = ord[i], j = i - 1; while (j >= 0 && adjn[ord[j]] > adjn[v]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = v; }
    for (int i = 0; i < n && budget > 0; i++) if (adjn[ord[i]]) { onpath[ord[i]] = 1; pathv[0] = ord[i]; lp(ord[i], 1); onpath[ord[i]] = 0; }
    free(ord); return bestpath;
}

static void analyze(const char *name, Set *S, int kmax, int *bases, int nbases, int do_graph) {
    printf("  [%s]\n", name);
    // 1) moldes
    MStat *st = calloc(1, sizeof(MStat)); long npairs = 0;
    for (int bi = 0; bi < nbases; bi++) {
        int s = bases[bi];
        for (int t = 0; t < S->n; t++) if (t != s) {
            int diff = 0; for (int i = 0; i < 4; i++) diff += __builtin_popcountll(S->v[s].w[i] ^ S->v[t].w[i]);
            if (diff > 2 * kmax) continue;
            npairs++; mold_chains(S, s, t, 0, st); mold_chains(S, s, t, 1, st);
        }
    }
    printf("    1) molde desplazado (parejas con ≤%d reinas cambiadas: %ld; 24 desplazamientos v):\n", kmax, npairs);
    for (int w = 0; w < 2; w++) {
        printf("       %s:", w ? "dando la vuelta" : "sin dar la vuelta"); int any = 0;
        for (int L = 3; L < 64; L++) if (st->hist[w][L]) { printf("  %d sol:%ld", L, st->hist[w][L]); any = 1; }
        if (!any) printf("  ninguna cadena de 3"); else printf("   | explicadas por D4: %ld sí / %ld no", st->expl[w], st->unexpl[w]);
        printf("\n");
        if (st->best[w] >= 3) {
            printf("       ejemplo sin explicar (%d soluciones, v = (%d,%d), %d reinas):\n", st->best[w], st->exv[w][0], st->exv[w][1], st->exk[w]);
            for (int a = 0; a < st->exn[w]; a++) { print_board(&st->ex[w][a]); }
            if (N >= 9) { printf("       W (pares de costura) de esas soluciones:"); for (int a = 0; a < st->exn[w]; a++) printf(" %d", seam_pairs(&st->ex[w][a])); printf("\n"); }
        }
    }
    free(st);
    if (!do_graph) return;
    // 2) saltos iguales
    int n = S->n; AD = 0;
    int (*E)[3] = malloc(sizeof(int) * 3 * 20000000); long ne = 0;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
        int diff = 0; for (int i = 0; i < 4; i++) diff += __builtin_popcountll(S->v[a].w[i] ^ S->v[b].w[i]);
        if (diff > 2 * MAXKD) continue;
        int d = bitdist(&S->v[a], &S->v[b]); if (d < 0 || ne >= 20000000) continue;
        E[ne][0] = a; E[ne][1] = b; E[ne][2] = d; ne++;
    }
    printf("    2) saltos iguales (aristas con ≤%d reinas cambiadas: %ld)\n", MAXKD, ne);
    adjn = calloc(n, sizeof(int)); onpath = calloc(n, sizeof(int)); pathv = calloc(256, sizeof(int));
    for (int d = 1; d <= 10; d++) {
        memset(adjn, 0, sizeof(int) * n); long m = 0;
        for (long i = 0; i < ne; i++) if (E[i][2] == d) { adjn[E[i][0]]++; adjn[E[i][1]]++; m++; }
        if (!m) continue;
        AD = 0; for (int i = 0; i < n; i++) if (adjn[i] > AD) AD = adjn[i];
        adj = malloc(sizeof(int) * (size_t)n * AD); memset(adjn, 0, sizeof(int) * n);
        int *eu = malloc(sizeof(int) * m), *ev = malloc(sizeof(int) * m); long q = 0;
        for (long i = 0; i < ne; i++) if (E[i][2] == d) { int u = E[i][0], v = E[i][1]; adj[u * AD + adjn[u]++] = v; adj[v * AD + adjn[v]++] = u; eu[q] = u; ev[q] = v; q++; }
        int Lr = longest(n, 20000000); int realbudget = budget > 0;
        int path[256]; memcpy(path, bestpv, sizeof path);
        // ¿se repite el molde (desplazado) a lo largo de la cadena?
        int same = 0;
        for (int i = 2; i < Lr && i < 256; i++) {
            const B *a0 = &S->v[path[i - 2]], *a1 = &S->v[path[i - 1]], *a2 = &S->v[path[i]];
            int r1[64], p1[64], r2[64], p2[64], k1 = 0, k2 = 0, k1b = 0, k2b = 0;
            for (int p = 0; p < NC; p++) { if (gb(a0, p) && !gb(a1, p)) r1[k1++] = p; if (gb(a1, p) && !gb(a0, p)) p1[k1b++] = p; if (gb(a1, p) && !gb(a2, p)) r2[k2++] = p; if (gb(a2, p) && !gb(a1, p)) p2[k2b++] = p; }
            if (k1 != k2 || k1 == 0) continue;
            int dr = r2[0] / N - r1[0] / N, dc = r2[0] % N - r1[0] % N, ok = 1;
            for (int j = 0; j < k1 && ok; j++) ok = r2[j] / N - r1[j] / N == dr && r2[j] % N - r1[j] % N == dc && p2[j] / N - p1[j] / N == dr && p2[j] % N - p1[j] % N == dc;
            same += ok;
        }
        // grafo al azar con los mismos grados (intercambio de aristas)
        for (long sw = 0; sw < 10 * m; sw++) {
            long i = rnd() % m, j = rnd() % m; int a = eu[i], b = ev[i], c = eu[j], e = ev[j];
            if (a == e || c == b || a == c || b == e) continue;
            int dup = 0; if (m <= 20000) for (long z = 0; z < m && !dup; z++) dup = (eu[z] == a && ev[z] == e) || (eu[z] == e && ev[z] == a) || (eu[z] == c && ev[z] == b) || (eu[z] == b && ev[z] == c);
            if (dup) continue;
            ev[i] = e; ev[j] = b;
            if (m > 4000 && sw > 4 * m) break;              // grafos grandes: menos intercambios
        }
        memset(adjn, 0, sizeof(int) * n);
        for (long i = 0; i < m; i++) { int u = eu[i], v = ev[i]; adj[u * AD + adjn[u]++] = v; adj[v * AD + adjn[v]++] = u; }
        int Lz = longest(n, 20000000); int zbudget = budget > 0;
        printf("       d=%2d: %6ld aristas   cadena más larga real %3d%s   al azar (mismos grados) %3d%s   tramos con el mismo molde desplazado: %d de %d\n",
               d, m, Lr, realbudget ? "" : "+", Lz, zbudget ? "" : "+", same, Lr > 2 ? Lr - 2 : 0);
        free(adj); free(eu); free(ev);
    }
    free(E); free(adjn); free(onpath); free(pathv);
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); NC = N * N; int kmax = argc > 2 ? atoi(argv[2]) : 4; const char *bm = argc > 3 ? argv[3] : "0";
    if (kmax > MAXK) kmax = MAXK;
    int nograph = getenv("NOGRAPH") != 0;
    set_init(&SOL, 400000); bt(0, 0, 0, 0);
    int *bases = malloc(sizeof(int) * SOL.n), nb = 0;
    for (int q = 0; q < SOL.n; q++) if (strcmp(bm, "w2") || seam_pairs(&SOL.v[q]) == 2) bases[nb++] = q;
    // falso: permutaciones al azar cerradas por D4
    Set Z; set_init(&Z, SOL.n + 16);
    while (Z.n < SOL.n) {
        int x[MAXN]; for (int i = 0; i < N; i++) x[i] = i;
        for (int i = N - 1; i > 0; i--) { int j = rnd() % (i + 1), t = x[i]; x[i] = x[j]; x[j] = t; }
        B b = {{0}}; for (int i = 0; i < N; i++) sb(&b, x[i] * N + i);
        if (set_find(&Z, &b) >= 0 || set_find(&SOL, &b) >= 0) continue;
        for (int g = 0; g < 8 && Z.n < Z.cap; g++) { B y; d4b(&b, g, &y); set_add(&Z, &y); }
    }
    int *zb = malloc(sizeof(int) * Z.n), nzb = 0;
    for (int q = 0; q < Z.n && nzb < nb; q++) zb[nzb++] = q;          // mismo nº de bases
    printf("N = %d   soluciones %d   bases %d%s   falso %d\n", N, SOL.n, nb, strcmp(bm, "w2") ? "" : " (casi toroidales, W=2)", Z.n);
    int graph = SOL.n <= 20000 && !nograph;
    analyze("real", &SOL, kmax, bases, nb, graph);
    analyze("falso", &Z, kmax, zb, nzb, graph);
    return 0;
}
