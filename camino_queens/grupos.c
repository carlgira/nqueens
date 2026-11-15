// grupos.c — generar muchos caminos y ver si alguno identifica un grupo.
//
// Un camino lineal es una bandera de subespacios afines F_0 ⊂ F_1 ⊂ … ⊂ F_K
// (cada nivel del árbol fija una cifra más). Las soluciones de F_k son las que
// caen en el primer subárbol de profundidad (dimensión − k).
//
// Generación codiciosa, partiendo de las soluciones: se elige una solución
// base al azar (F_0); en cada paso se proyectan las demás módulo las
// direcciones de F_{k-1} y se elige la dirección que más soluciones añade.
// n_k = soluciones en F_k.
//
// Para no engañarse: el mismo proceso, con el mismo número de caminos, sobre
// un conjunto falso del mismo tamaño: permutaciones al azar con Σ i·x ≡ Σ i²·x ≡ 0
// (las sumas que cumplen todas las toroidales), cerrado por las mismas
// simetrías afines (desplazar columnas, escala (i,x)->(ui,ux), x->-x).
// Todo se trabaja con x_0 = 0 (cociente por subir/bajar todas las reinas).
//
// Uso: ./grupos p clase caminos [semilla]
//      clase: c = con centro, o = otra simetría, n = sin simetría, t = todas

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXP 32
#define KMAX 24
static int P, inv[MAXP];
static int md(int a) { a %= P; return a < 0 ? a + P : a; }

static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

typedef unsigned char row[MAXP];
typedef struct { int n, cap; row *v; } Set;
static void push(Set *S, const int *x) {
    if (S->n == S->cap) { S->cap = S->cap ? 2 * S->cap : 4096; S->v = realloc(S->v, S->cap * sizeof(row)); }
    for (int i = 0; i < P; i++) S->v[S->n][i] = x[i]; S->n++;
}

// ---------- simetrías (como en simetria.c) ----------
static int G[64][4], nG;
static void norm(int *m) { int f = m[0] ? m[0] : m[1]; int s = inv[f]; for (int k = 0; k < 4; k++) m[k] = md(m[k] * s); }
static void mul(const int *a, const int *b, int *c) {
    c[0] = md(a[0] * b[0] + a[1] * b[2]); c[1] = md(a[0] * b[1] + a[1] * b[3]);
    c[2] = md(a[2] * b[0] + a[3] * b[2]); c[3] = md(a[2] * b[1] + a[3] * b[3]);
}
static void build_group(void) {
    int gens[3][4] = {{0, 1, 1, 0}, {1, 0, 0, P - 1}, {1, 1, P - 1, 1}};
    nG = 1; G[0][0] = 1; G[0][1] = 0; G[0][2] = 0; G[0][3] = 1;
    for (int a = 0; a < nG; a++) for (int g = 0; g < 3; g++) {
        int c[4]; mul(gens[g], G[a], c); norm(c);
        int seen = 0; for (int b = 0; b < nG; b++) if (!memcmp(G[b], c, sizeof c)) seen = 1;
        if (!seen) memcpy(G[nG++], c, sizeof c);
    }
}
static int has_sym(const int *x, int g, int s) {
    int y[MAXP]; const int *m = G[g];
    for (int i = 0; i < P; i++) y[md(s * (m[0] * i + m[1] * x[i]))] = md(s * (m[2] * i + m[3] * x[i]));
    for (int t1 = 0; t1 < P; t1++) {
        if (g == 0 && s == 1 && t1 == 0) continue;
        int t2 = md(x[0] - y[md(-t1)]), ok = 1;
        for (int i = 1; i < P && ok; i++) ok = x[i] == md(y[md(i - t1)] + t2);
        if (ok) return 1;
    }
    return 0;
}
static char classify(const int *x) {
    int other = 0;
    for (int g = 0; g < nG; g++) for (int s = 1; s < P; s++) {
        if (g == 0 && s == 1) continue;
        if (has_sym(x, g, s)) { if (G[g][1] == 0 && G[g][2] == 0) return 'c'; other = 1; }
    }
    return other ? 'o' : 'n';
}

// ---------- toroidales con x_0 = 0 ----------
static Set REAL; static char want;
static int xs[MAXP];
static void bt(int c, unsigned rows, unsigned dg, unsigned ad) {
    if (c == P) { if (want == 't' || classify(xs) == want) push(&REAL, xs); return; }
    for (int r = 0; r < P; r++) {
        unsigned a = 1u << r, b = 1u << md(r - c), e = 1u << md(r + c);
        if ((rows & a) || (dg & b) || (ad & e)) continue;
        xs[c] = r; bt(c + 1, rows | a, dg | b, ad | e);
    }
}

// ---------- conjunto de puntos con búsqueda por hash ----------
static uint64_t hrow(const unsigned char *v) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < P; i++) h = (h ^ v[i]) * 1099511628211ULL; return h; }
typedef struct { int cap; int *slot; } Index;      // índices a Set, -1 vacío
static int idx_find(Index *I, Set *S, const int *x) {
    unsigned char v[MAXP]; for (int i = 0; i < P; i++) v[i] = x[i];
    uint64_t k = hrow(v) & (I->cap - 1);
    while (I->slot[k] >= 0) { if (!memcmp(S->v[I->slot[k]], v, P)) return 1; k = (k + 1) & (I->cap - 1); }
    return 0;
}
static void idx_add(Index *I, Set *S, const int *x) {
    unsigned char v[MAXP]; for (int i = 0; i < P; i++) v[i] = x[i];
    uint64_t k = hrow(v) & (I->cap - 1);
    while (I->slot[k] >= 0) { if (!memcmp(S->v[I->slot[k]], v, P)) return; k = (k + 1) & (I->cap - 1); }
    push(S, x); I->slot[k] = S->n - 1;
}

// Conjunto falso: órbitas de permutaciones al azar con los momentos a 0.
static void normalize0(int *x) { int c = x[0]; for (int i = 0; i < P; i++) x[i] = md(x[i] - c); }
static void build_null(Set *Z, int target) {
    Index I; I.cap = 1; while (I.cap < 4 * target) I.cap <<= 1;
    I.slot = malloc(I.cap * sizeof(int)); memset(I.slot, -1, I.cap * sizeof(int));
    int x[MAXP], y[MAXP], z[MAXP];
    while (Z->n < target) {
        for (int i = 0; i < P; i++) x[i] = i;
        for (int i = P - 1; i > 0; i--) { int j = rnd() % (i + 1), t = x[i]; x[i] = x[j]; x[j] = t; }
        long m1 = 0, m2 = 0; for (int i = 0; i < P; i++) { m1 += (long)i * x[i]; m2 += (long)i * i % P * x[i]; }
        if (m1 % P || m2 % P) continue;
        normalize0(x);
        if (idx_find(&I, Z, x)) continue;
        for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++) {
            for (int i = 0; i < P; i++) y[md(u * i)] = md(u * x[i] * (sg ? -1 : 1));
            for (int i = 0; i < P; i++) z[i] = y[(i + t) % P];
            normalize0(z); idx_add(&I, Z, z);
        }
    }
    free(I.slot);
    // recortar al tamaño exacto (submuestra al azar)
    for (int i = 0; i < target; i++) { int j = i + rnd() % (Z->n - i); row t; memcpy(t, Z->v[i], sizeof t); memcpy(Z->v[i], Z->v[j], sizeof t); memcpy(Z->v[j], t, sizeof t); }
    Z->n = target;
}

// ---------- caminos codiciosos ----------
typedef struct { int cap; unsigned char (*key)[MAXP]; int *cnt; char *used; } Tab;
static void tab_init(Tab *T, int n) { T->cap = 1; while (T->cap < 2 * n) T->cap <<= 1; T->key = malloc(T->cap * MAXP); T->cnt = malloc(T->cap * sizeof(int)); T->used = malloc(T->cap); }
static int tab_add(Tab *T, const unsigned char *k) {
    uint64_t h = hrow(k) & (T->cap - 1);
    while (T->used[h] && memcmp(T->key[h], k, P)) h = (h + 1) & (T->cap - 1);
    if (!T->used[h]) { T->used[h] = 1; memcpy(T->key[h], k, P); T->cnt[h] = 0; }
    return ++T->cnt[h], (int)h;
}

// Devuelve n_k (k = 0..K) en prof[]; dirs en B.
static void camino(Set *S, int base, int K, int *prof, Tab *T) {
    int B[KMAX][MAXP], piv[KMAX], r = 0;
    int nin = 1; prof[0] = 1;
    for (int k = 1; k <= K; k++) {
        memset(T->used, 0, T->cap);
        int best = 0, bestn = 0;
        for (int q = 0; q < S->n; q++) {
            int w[MAXP]; for (int j = 0; j < P; j++) w[j] = md(S->v[q][j] - S->v[base][j]);
            for (int a = 0; a < r; a++) { int c = w[piv[a]]; if (c) for (int j = 0; j < P; j++) w[j] = md(w[j] - c * B[a][j]); }
            int f = -1; for (int j = 0; j < P; j++) if (w[j]) { f = j; break; }
            if (f < 0) continue;                                // ya está en F_{k-1}
            int s = inv[w[f]]; unsigned char key[MAXP]; for (int j = 0; j < P; j++) key[j] = md(w[j] * s);
            int h = tab_add(T, key);
            int c = T->cnt[h];
            if (c > bestn) { bestn = c; best = h; }
        }
        // desempate uniforme entre todas las direcciones con el máximo (antes se
        // favorecían las últimas del array, que en el real son la misma órbita)
        if (bestn > 0) { int ties = 0; for (int h2 = 0; h2 < T->cap; h2++) if (T->used[h2] && T->cnt[h2] == bestn && rnd() % ++ties == 0) best = h2; }
        if (bestn == 0) { for (; k <= K; k++) prof[k] = nin; return; }
        // añadir dirección (forma escalonada reducida)
        int v[MAXP]; for (int j = 0; j < P; j++) v[j] = T->key[best][j];
        int f = -1; for (int j = 0; j < P; j++) if (v[j]) { f = j; break; }
        for (int a = 0; a < r; a++) { int c = B[a][f]; if (c) for (int j = 0; j < P; j++) B[a][j] = md(B[a][j] - c * v[j]); }
        memcpy(B[r], v, sizeof v); piv[r] = f; r++;
        nin += bestn; prof[k] = nin;
    }
}

static void run(const char *name, Set *S, int ncam, int K, long *maxk, double *meank, int (*bestprof)[KMAX + 1], int *bestbase) {
    for (int i = S->n - 1; i > 0; i--) { int j = rnd() % (i + 1); row t; memcpy(t, S->v[i], sizeof t); memcpy(S->v[i], S->v[j], sizeof t); memcpy(S->v[j], t, sizeof t); }
    Tab T; tab_init(&T, S->n);
    int prof[KMAX + 1];
    for (int k = 0; k <= K; k++) { maxk[k] = 0; meank[k] = 0; }
    for (int c = 0; c < ncam; c++) {
        int base = rnd() % S->n;
        camino(S, base, K, prof, &T);
        for (int k = 0; k <= K; k++) { meank[k] += prof[k]; if (prof[k] > maxk[k]) { maxk[k] = prof[k]; bestbase[k] = base; memcpy(bestprof[k], prof, sizeof prof); } }
    }
    for (int k = 0; k <= K; k++) meank[k] /= ncam;
    (void)name;
    free(T.key); free(T.cnt); free(T.used);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: ./grupos p clase caminos [semilla]\n"); return 1; }
    P = atoi(argv[1]); want = argv[2][0]; int ncam = atoi(argv[3]);
    if (argc > 4) rs ^= (uint64_t)atoll(argv[4]) * 0x9E3779B97F4A7C15ULL;
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_group();
    xs[0] = 0; bt(1, 1u, 1u, 1u);
    Set Z = {0, 0, 0}; build_null(&Z, REAL.n);
    int K = P - 4; if (K > KMAX) K = KMAX;            // dimensión útil: p-1 (x_0=0) menos 3 momentos
    printf("p = %d   clase %c   soluciones (x_0=0) %d   falso %d   caminos %d\n", P, want, REAL.n, Z.n, ncam);
    long mr[KMAX + 1], mz[KMAX + 1]; double ar[KMAX + 1], az[KMAX + 1];
    int bpr[KMAX + 1][KMAX + 1], bpz[KMAX + 1][KMAX + 1], bbr[KMAX + 1], bbz[KMAX + 1];
    run("real", &REAL, ncam, K, mr, ar, bpr, bbr);
    run("falso", &Z, ncam, K, mz, az, bpz, bbz);
    printf("  k   tableros del subárbol   real: máx  media     falso: máx  media    máx real / máx falso\n");
    for (int k = 1; k <= K; k++)
        printf("  %2d   p^%-2d                   %8ld %7.1f    %8ld %7.1f     %.2f\n", k, k, mr[k], ar[k], mz[k], az[k], (double)mr[k] / mz[k]);
    // el mejor camino real a la mitad de la profundidad
    int kk = K / 2; printf("  mejor camino real en k=%d (base %d): perfil", kk, bbr[kk]);
    for (int k = 0; k <= K; k++) printf(" %d", bpr[kk][k]);
    printf("\n  base:"); for (int i = 0; i < P; i++) printf(" %d", REAL.v[bbr[kk]][i]); printf("\n");
    return 0;
}
