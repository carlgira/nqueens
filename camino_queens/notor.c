// notor.c — trayectorias entre soluciones NO toroidales (tablero normal, sin dar la vuelta).
//
// En el tablero normal las cuentas son enteras: no hay "mod N". Familias:
//   recta        b_k = s + k·v, v entero primitivo         (la fijan 2; cuenta desde la 3.ª)
//   parábola     b_k = s + k·v + C(k,2)·w, w entero          (la fijan 3; desde la 4.ª)
//   diagonal     el salto de cada columna se multiplica por λ_i racional en cada paso
//                                                             (la fijan 3; desde la 4.ª)
//   permutación  G(x) = σ∘x (renombrar filas). Es la misma cadena que permutar
//                columnas: s∘(s⁻¹t)^k = (t s⁻¹)^k∘s.             (la fijan 2; desde la 3.ª)
//                Se separa en: σ giro cíclico de filas (r -> r+c mod N), π giro
//                cíclico de columnas (i -> i+c mod N), y el resto.
// Las soluciones intermedias de una trayectoria no cuentan; solo los aciertos.
// Se excluyen las G que son simetrías (π o σ = identidad o espejo).
//
// Simetrías del tablero normal: D4 (8). Las 4 sin transponer son afines y
// conjugan cada familia en sí misma: basta una base por órbita de ellas.
// Cada cadena se marca "explicada" si algún par de sus soluciones está
// relacionado por D4.
// Falso: permutaciones al azar en órbitas completas por D4, tantas como las
// soluciones. Las bases son soluciones no toroidales; los aciertos cuentan en
// todas las soluciones.
//
// Uso: ./notor N bases pares_muestreados [semilla]
//      bases: nº de bases (0 = todas); pares: para diagonal y parábola

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN 16
typedef unsigned char row[MAXN];
static int N;
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int iabs(int a) { return a < 0 ? -a : a; }
static int gcd(int a, int b) { a = iabs(a); b = iabs(b); while (b) { int t = a % b; a = b; b = t; } return a; }

// ---------- conjunto con hash ----------
typedef struct { int n, cap, hcap; row *v; int *slot; char *tor; } Big;
static uint64_t hrow(const unsigned char *v) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < N; i++) h = (h ^ v[i]) * 1099511628211ULL; return h ^ (h >> 29); }
static void big_init(Big *S, int cap) {
    S->n = 0; S->cap = cap; S->v = calloc(cap, sizeof(row)); S->tor = calloc(cap, 1);
    S->hcap = 1; while (S->hcap < 2 * cap) S->hcap <<= 1;
    S->slot = malloc(S->hcap * sizeof(int)); memset(S->slot, -1, S->hcap * sizeof(int));
}
static int big_find(const Big *S, const unsigned char *x) {
    uint64_t k = hrow(x) & (S->hcap - 1);
    while (S->slot[k] >= 0) { if (!memcmp(S->v[S->slot[k]], x, N)) return S->slot[k]; k = (k + 1) & (S->hcap - 1); }
    return -1;
}
static int big_add(Big *S, const unsigned char *x) {
    uint64_t k = hrow(x) & (S->hcap - 1);
    while (S->slot[k] >= 0) { if (!memcmp(S->v[S->slot[k]], x, N)) return -1; k = (k + 1) & (S->hcap - 1); }
    if (S->n == S->cap) { S->cap *= 2; S->v = realloc(S->v, S->cap * sizeof(row)); S->tor = realloc(S->tor, S->cap); }
    memset(S->v[S->n], 0, sizeof(row)); memcpy(S->v[S->n], x, N); S->tor[S->n] = 0;
    // rehash si hace falta
    S->slot[k] = S->n++;
    if (2 * S->n > S->hcap) {
        S->hcap <<= 1; S->slot = realloc(S->slot, S->hcap * sizeof(int)); memset(S->slot, -1, S->hcap * sizeof(int));
        for (int q = 0; q < S->n; q++) { uint64_t h = hrow(S->v[q]) & (S->hcap - 1); while (S->slot[h] >= 0) h = (h + 1) & (S->hcap - 1); S->slot[h] = q; }
    }
    return S->n - 1;
}

// ---------- soluciones ----------
static Big SOL; static int xs[MAXN];
static void bt(int c, unsigned rows, unsigned d1, unsigned d2) {
    if (c == N) { unsigned char y[MAXN]; for (int i = 0; i < N; i++) y[i] = xs[i]; big_add(&SOL, y); return; }
    for (int r = 0; r < N; r++) {
        unsigned a = 1u << r, b = 1u << (r + c), e = 1u << (r - c + N);
        if ((rows & a) || (d1 & b) || (d2 & e)) continue;
        xs[c] = r; bt(c + 1, rows | a, d1 | b, d2 | e);
    }
}
static int is_sol(const unsigned char *x) {         // por si hace falta comprobar directamente
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++)
        if (x[i] == x[j] || iabs(x[i] - x[j]) == j - i) return 0;
    return 1;
}
static int is_tor(const unsigned char *x) {
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        int d = ((x[j] - x[i]) % N + N) % N, k = j - i;
        if (d == k % N || d == (N - k) % N) return 0;
    }
    return 1;
}
// D4: g = (transponer, espejo columnas, espejo filas)
static void d4(const unsigned char *x, int g, unsigned char *y) {
    for (int i = 0; i < N; i++) {
        int a = i, b = x[i];
        if (g & 1) { int t = a; a = b; b = t; }
        if (g & 2) a = N - 1 - a;
        if (g & 4) b = N - 1 - b;
        y[a] = b;
    }
}
static int related(const unsigned char *a, const unsigned char *b) {
    unsigned char y[MAXN];
    for (int g = 0; g < 8; g++) { d4(a, g, y); if (!memcmp(y, b, N)) return 1; }
    return 0;
}

// ---------- estadística ----------
enum { RECTA, PARAB, DIAG, PGIROF, PGIROC, PAFIN, POTRA, NF };
static const char *FN[NF] = {"recta", "parábola", "diagonal (λ_i)", "girar filas (r+c mod N)", "girar columnas (i+c)", "afín mod N (a·r+c, a≠1)", "otra permutación"};
static const int FIT[NF] = {2, 3, 3, 2, 2, 2, 2};
typedef struct { long hist[NF][64], expl[NF], unexpl[NF], torhit[NF], allcorner[NF], somecorner[NF], longc[NF], longcorner[NF]; int best[NF]; row ex[NF][6]; int exn[NF]; } Stat;
// ¿x es una toroidal de N+1 sin la reina de una esquina? (en alguna de las 8 orientaciones)
static int from_torus(const unsigned char *x) {
    unsigned char y[MAXN];
    for (int g = 0; g < 8; g++) {
        d4(x, g, y);
        int M = N + 1, ok = 1, z[MAXN + 1];
        for (int i = 0; i < N; i++) z[i] = y[i];
        z[N] = N;
        for (int i = 0; i <= N && ok; i++) for (int j = i + 1; j <= N && ok; j++) {
            int d = ((z[j] - z[i]) % M + M) % M, k = j - i;
            if (d == 0 || d == k % M || d == (M - k) % M) ok = 0;
        }
        if (ok) return 1;
    }
    return 0;
}
static char *CORNER;
static Big *CUR;
static void record(Stat *st, int f, int *idx, int L) {
    st->hist[f][L < 63 ? L : 63]++;
    int e = 0, th = 0;
    for (int a = 0; a < L && !e; a++) for (int b = a + 1; b < L && !e; b++) e = related(CUR->v[idx[a]], CUR->v[idx[b]]);
    for (int a = 0; a < L; a++) th |= CUR->tor[idx[a]];
    if (CORNER) { int all = 1, some = 0; for (int a = 0; a < L; a++) { all &= CORNER[idx[a]]; some |= CORNER[idx[a]]; }
        st->allcorner[f] += all; st->somecorner[f] += some;
        if (L >= 5) { st->longc[f]++; st->longcorner[f] += all; } }
    if (e) st->expl[f]++; else st->unexpl[f]++;
    st->torhit[f] += th;
    if (L > st->best[f]) { st->best[f] = L; st->exn[f] = L < 6 ? L : 6; for (int a = 0; a < st->exn[f]; a++) memcpy(st->ex[f][a], CUR->v[idx[a]], sizeof(row)); }
}
static int chain_add(int *idx, int L, int r) { for (int a = 0; a < L; a++) if (idx[a] == r) return 0; idx[L] = r; return 1; }

// ---------- rectas por una base ----------
typedef struct { uint64_t h; int q; } Ent;
static int cmp_ent(const void *a, const void *b) { uint64_t x = ((const Ent *)a)->h, y = ((const Ent *)b)->h; return x < y ? -1 : x > y; }
static void lines(Big *S, int sb, Ent *E, Stat *st) {
    const unsigned char *s = S->v[sb]; int m = 0;
    for (int q = 0; q < S->n; q++) if (q != sb) {
        int v[MAXN], g = 0, f = -1;
        for (int i = 0; i < N; i++) { v[i] = S->v[q][i] - s[i]; g = gcd(g, v[i]); if (f < 0 && v[i]) f = i; }
        int sg = v[f] < 0 ? -1 : 1; unsigned char key[MAXN];
        for (int i = 0; i < N; i++) key[i] = (unsigned char)(sg * v[i] / g + 64);
        E[m].h = hrow(key); E[m].q = q; m++;
    }
    qsort(E, m, sizeof(Ent), cmp_ent);
    for (int i = 0; i < m;) {
        int j = i; while (j < m && E[j].h == E[i].h) j++;
        if (j - i >= 2) { int idx[64], L = 0; idx[L++] = sb; for (int a = i; a < j && L < 63; a++) idx[L++] = E[a].q; record(st, RECTA, idx, L); }
        i = j;
    }
}

// ---------- parábolas enteras: s (k=0), t (k=1), q (k) ----------
static void parabolas(Big *S, int sb, int tb, Ent *E, Stat *st) {
    const unsigned char *s = S->v[sb], *t = S->v[tb]; int v[MAXN]; long m = 0;
    for (int i = 0; i < N; i++) v[i] = t[i] - s[i];
    for (int q = 0; q < S->n; q++) {
        if (q == sb || q == tb) continue;
        for (int k = 2; k <= N; k++) {
            int den = k * (k - 1) / 2, w[MAXN], ok = 1, nz = 0;
            for (int i = 0; i < N && ok; i++) { int num = S->v[q][i] - s[i] - k * v[i]; if (num % den) ok = 0; else { w[i] = num / den; nz |= w[i]; } }
            if (!ok || !nz) continue;
            // degenerada si w ∥ v
            int par = 1; for (int i = 0; i < N && par; i++) for (int j = i + 1; j < N && par; j++) par = w[i] * v[j] == w[j] * v[i];
            if (par) continue;
            unsigned char key[MAXN]; for (int i = 0; i < N; i++) key[i] = (unsigned char)(w[i] + 128);
            E[m].h = hrow(key) ^ (uint64_t)0; E[m].q = q; m++;
        }
    }
    qsort(E, m, sizeof(Ent), cmp_ent);
    for (long i = 0; i < m;) {
        long j = i; while (j < m && E[j].h == E[i].h) j++;
        if (j - i >= 2) { int idx[64], L = 0; idx[L++] = sb; idx[L++] = tb; for (long a = i; a < j && L < 63; a++) if (chain_add(idx, L, E[a].q)) L++; if (L >= 4) record(st, PARAB, idx, L); }
        i = j;
    }
}

// ---------- diagonal: el salto de la columna i se multiplica por λ_i = d2/d1 ----------
static void diag_chains(Big *S, int sb, int tb, Stat *st) {
    const unsigned char *s = S->v[sb], *t = S->v[tb];
    for (int ub = 0; ub < S->n; ub++) {
        if (ub == sb || ub == tb) continue;
        const unsigned char *u = S->v[ub]; int num[MAXN], den[MAXN], d[MAXN], ok = 1, all1 = 1;
        for (int i = 0; i < N && ok; i++) {
            int d1 = t[i] - s[i], d2 = u[i] - t[i];
            if (!d1) { if (d2) ok = 0; num[i] = 1; den[i] = 1; d[i] = 0; continue; }
            int g = gcd(d1, d2); if (!g) g = 1;
            num[i] = d2 / g; den[i] = d1 / g; if (den[i] < 0) { den[i] = -den[i]; num[i] = -num[i]; }
            if (num[i] != den[i]) all1 = 0;
            d[i] = d2;
        }
        if (!ok || all1) continue;
        int idx[64], L = 3; idx[0] = sb; idx[1] = tb; idx[2] = ub; unsigned char cur[MAXN]; memcpy(cur, u, N);
        for (;;) {
            unsigned char nx[MAXN]; int good = 1;
            for (int i = 0; i < N && good; i++) {
                long nd = (long)d[i] * num[i];
                if (nd % den[i]) { good = 0; break; }
                nd /= den[i]; int val = cur[i] + (int)nd;
                if (val < 0 || val >= N) { good = 0; break; }
                nx[i] = val; d[i] = (int)nd;
            }
            if (!good) break;
            int r = big_find(S, nx); if (r < 0 || L >= 63 || !chain_add(idx, L, r)) break;
            L++; memcpy(cur, nx, N);
        }
        if (L >= 4) record(st, DIAG, idx, L);
    }
}

// ---------- permutar columnas / renombrar filas ----------
static void perm_chains(Big *S, int sb, int tb, Stat *st) {
    const unsigned char *s = S->v[sb], *t = S->v[tb]; int sinv[MAXN], pi[MAXN], sig[MAXN];
    for (int i = 0; i < N; i++) sinv[s[i]] = i;
    for (int i = 0; i < N; i++) pi[i] = sinv[t[i]];            // t = s∘π
    for (int r = 0; r < N; r++) sig[r] = t[sinv[r]];           // t = σ∘s (misma cadena)
    int idn = 1, rev = 1; for (int r = 0; r < N; r++) { idn &= sig[r] == r; rev &= sig[r] == N - 1 - r; }
    if (idn || rev) return;
    int rc = 1, cc = 1, c1 = sig[0], c2 = pi[0];
    for (int r = 0; r < N; r++) rc &= sig[r] == (r + c1) % N;
    for (int i = 0; i < N; i++) cc &= pi[i] == (i + c2) % N;
    int af = 0;                                   // σ o π afín mod N con a ≠ 1
    for (int a = 2; a < N && !af; a++) {
        int ok = 1; for (int r = 0; r < N && ok; r++) ok = sig[r] == (a * r + c1) % N; if (ok) af = 1;
        ok = 1; for (int i = 0; i < N && ok; i++) ok = pi[i] == (a * i + c2) % N; if (ok) af = 1;
    }
    int f = rc ? PGIROF : cc ? PGIROC : af ? PAFIN : POTRA;
    int idx[64], L = 2; idx[0] = sb; idx[1] = tb; unsigned char cur[MAXN]; memcpy(cur, t, N);
    for (;;) { unsigned char nx[MAXN]; for (int i = 0; i < N; i++) nx[i] = sig[cur[i]];
        int r = big_find(S, nx); if (r < 0 || L >= 63 || !chain_add(idx, L, r)) break; L++; memcpy(cur, nx, N); }
    if (L >= 3) record(st, f, idx, L);
}

static void analyze(const char *name, Big *S, int nbases, int npairs, int only_notor) {
    CUR = S;
    CORNER = calloc(S->n, 1); int ncor = 0;
    for (int q = 0; q < S->n; q++) { CORNER[q] = from_torus(S->v[q]); ncor += CORNER[q]; }
    printf("  (%s: %d de %d son toroidal de N+1 menos una esquina)\n", name, ncor, S->n);
    // bases: una por órbita de las 4 simetrías afines, entre las no toroidales
    char *seen = calloc(S->n, 1); int *bases = malloc(S->n * sizeof(int)), nb = 0;
    for (int q = 0; q < S->n; q++) if (!seen[q]) {
        if (!only_notor || !S->tor[q]) bases[nb++] = q;
        unsigned char y[MAXN]; for (int g = 0; g < 8; g += 2) { d4(S->v[q], g, y); int r = big_find(S, y); if (r >= 0) seen[r] = 1; }
    }
    for (int i = nb - 1; i > 0; i--) { int j = rnd() % (i + 1), t = bases[i]; bases[i] = bases[j]; bases[j] = t; }
    if (nbases > 0 && nbases < nb) nb = nbases;
    Stat *st = calloc(1, sizeof(Stat)); Ent *E = malloc((size_t)S->n * (N + 1) * sizeof(Ent));
    for (int b = 0; b < nb; b++) {
        lines(S, bases[b], E, st);
        for (int t = 0; t < S->n; t++) if (t != bases[b]) perm_chains(S, bases[b], t, st);
        for (int k = 0; k < npairs; k++) { int t = rnd() % S->n; if (t == bases[b]) continue; diag_chains(S, bases[b], t, st); parabolas(S, bases[b], t, E, st); }
    }
    printf("  [%s]  bases %d (pares muestreados por base: %d)\n", name, nb, npairs);
    for (int f = 0; f < NF; f++) {
        printf("    %-18s (la fijan %d):", FN[f], FIT[f]); int any = 0;
        for (int L = FIT[f] + 1; L < 64; L++) if (st->hist[f][L]) { printf("  %d sol:%ld", L, st->hist[f][L]); any = 1; }
        if (!any) printf("  ninguna");
        else { printf("   | explicadas por D4: %ld sí / %ld no;  tocan alguna toroidal: %ld", st->expl[f], st->unexpl[f], st->torhit[f]);
               if (CORNER) printf(";  toroidal(N+1)−esquina: todas %ld, alguna %ld; de las de ≥5: %ld de %ld", st->allcorner[f], st->somecorner[f], st->longcorner[f], st->longc[f]); }
        printf("\n");
    }
    for (int f = 0; f < NF; f++) if (st->best[f] > FIT[f] && st->unexpl[f]) {
        printf("    ejemplo (%s, %d soluciones):\n", FN[f], st->best[f]);
        for (int a = 0; a < st->exn[f]; a++) { printf("     "); for (int i = 0; i < N; i++) printf(" %2d", st->ex[f][a][i]); printf("\n"); }
    }
    free(st); free(E); free(seen); free(bases); free(CORNER); CORNER = 0;
}

// Para cada solución: C(s) = {c ≠ 0 : s + c (mod N) también es solución}, y la
// racha más larga de giros consecutivos k·c (la trayectoria s, s+c, s+2c, …).
static void giros(const char *name, Big *S) {
    long hc[MAXN + 1] = {0}, hr[MAXN + 1] = {0};
    for (int q = 0; q < S->n; q++) {
        if (S->tor[q]) continue;
        int okc[MAXN] = {0}, cnt = 0;
        for (int c = 1; c < N; c++) { unsigned char y[MAXN]; for (int i = 0; i < N; i++) y[i] = (S->v[q][i] + c) % N; okc[c] = big_find(S, y) >= 0; cnt += okc[c]; }
        okc[0] = 1;
        int best = 1;                                 // nº de soluciones en la mejor trayectoria s + k·c
        for (int c = 1; c < N; c++) { int per = N / gcd(c, N), L = 1; while (L < per && okc[(L * c) % N]) L++; if (L > best) best = L; }
        hc[cnt]++; hr[best]++;
        if (getenv("SHOW") && best >= atoi(getenv("SHOW")) && hr[best] <= 2) {
            int bc = 0; for (int c = 1; c < N; c++) { int per = N / gcd(c, N), L = 1; while (L < per && okc[(L * c) % N]) L++; if (L == best) { bc = c; break; } }
            printf("    ejemplo: %d soluciones con c = %d\n", best, bc);
            for (int k = 0; k < best; k++) { printf("     "); for (int i = 0; i < N; i++) printf(" %2d", (S->v[q][i] + k * bc) % N); printf("\n"); }
        }
    }
    printf("  [%s] giros de filas que siguen siendo solución (nº de c) →", name);
    for (int k = 0; k <= N; k++) if (hc[k]) printf("  %d:%ld", k, hc[k]);
    printf("\n        trayectoria s, s+c, s+2c… más larga (soluciones) →");
    for (int k = 1; k <= N; k++) if (hr[k]) printf("  %d:%ld", k, hr[k]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: ./notor N bases pares [semilla]\n"); return 1; }
    N = atoi(argv[1]); int nbases = atoi(argv[2]), npairs = atoi(argv[3]);
    if (argc > 4) rs ^= (uint64_t)atoll(argv[4]) * 0x9E3779B97F4A7C15ULL;
    big_init(&SOL, 1 << 16); bt(0, 0, 0, 0);
    int ntor = 0; for (int q = 0; q < SOL.n; q++) { SOL.tor[q] = is_tor(SOL.v[q]); ntor += SOL.tor[q]; }
    // falso: permutaciones al azar en órbitas por D4, tantas como soluciones
    Big Z; big_init(&Z, SOL.n + 16);
    while (Z.n < SOL.n) {
        unsigned char x[MAXN]; for (int i = 0; i < N; i++) x[i] = i;
        for (int i = N - 1; i > 0; i--) { int j = rnd() % (i + 1), t = x[i]; x[i] = x[j]; x[j] = t; }
        if (big_find(&Z, x) >= 0 || is_sol(x)) continue;
        unsigned char y[MAXN]; for (int g = 0; g < 8; g++) { d4(x, g, y); big_add(&Z, y); }
    }
    printf("N = %d   soluciones %d (toroidales %d, no toroidales %d)   falso %d\n", N, SOL.n, ntor, SOL.n - ntor, Z.n);
    if (getenv("GIROS")) { giros("real (no toroidales)", &SOL); giros("falso", &Z); return 0; }
    analyze("real: bases no toroidales, aciertos en todas", &SOL, nbases, npairs, 1);
    analyze("falso", &Z, nbases, npairs, 0);
    return 0;
}
