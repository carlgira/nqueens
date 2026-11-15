// grupos3.c — ¿el exceso de las asimétricas sobrevive a un falso con las mismas simetrías?
//
// Igual que grupos2.c (caminos codiciosos sobre momentos M_{j,l} = Σ c^j x_c^l),
// pero con el conjunto completo (todas las filas b, sin normalizar x_0) y un
// falso cerrado por las mismas simetrías. El grupo H (desplazar columnas t,
// subir filas b, escala (i,x)->(ui,ux), x->-x) actúa LINEALMENTE sobre los
// momentos:
//   columnas t: M'_{j,l} = Σ_m C(j,m)(-t)^{j-m} M_{m,l}
//   filas b:    M'_{j,l} = Σ_m C(l,m) b^{l-m} M_{j,m}      (M_{j,0} = Σ c^j)
//   escala u:   M'_{j,l} = u^{j+l} M_{j,l};   signo: (-1)^l M_{j,l}
// Falso con simetrías: tantos puntos al azar en la envoltura de φ(reales)
// como órbitas tienen las reales, y de cada uno su órbita completa por H.
//
// Uso: ./grupos3 p clase lectura parámetro caminos K [semilla]   (lectura: sep | tot)

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

// ---------- lecturas ----------
#define MAXD 512
static int D, ftype, fj[MAXD], fl[MAXD], pa[MAXD], pb[MAXD], pw[MAXP][MAXP];
static void build_reading(const char *mode, int par) {
    for (int a = 0; a < P; a++) { pw[a][0] = 1; for (int e = 1; e < P; e++) pw[a][e] = pw[a][e - 1] * a % P; }
    D = 0;
    if (!strcmp(mode, "sep")) { ftype = 0; for (int l = 1; l <= par; l++) for (int j = 0; j < P; j++) { fj[D] = j; fl[D] = l; D++; } }
    else if (!strcmp(mode, "tot")) { ftype = 0; for (int l = 1; l <= par; l++) for (int j = 0; j + l <= par && j < P; j++) { fj[D] = j; fl[D] = l; D++; } }
    else if (!strcmp(mode, "quad")) { ftype = 1; for (int a = 0; a < P; a++) { pa[D] = a; pb[D] = -1; D++; } for (int a = 0; a < P; a++) for (int b = a; b < P; b++) { pa[D] = a; pb[D] = b; D++; } }
    else { fprintf(stderr, "lectura desconocida\n"); exit(1); }
    if (D > MAXD) { fprintf(stderr, "D = %d demasiado grande\n", D); exit(1); }
}
static void feat(const unsigned char *x, unsigned char *f) {
    for (int d = 0; d < D; d++) {
        long s = 0;
        if (ftype == 0) { for (int c = 0; c < P; c++) s += pw[c][fj[d]] * pw[x[c]][fl[d]]; }
        else s = pb[d] < 0 ? x[pa[d]] : x[pa[d]] * x[pb[d]];
        f[d] = s % P;
    }
}

// ---------- tabla de claves de longitud D ----------
static uint64_t hkey(const unsigned char *v) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < D; i++) h = (h ^ v[i]) * 1099511628211ULL; return h; }
typedef struct { int cap; unsigned char *key; int *cnt; char *used; } Tab;
static void tab_init(Tab *T, int n) { T->cap = 1; while (T->cap < 2 * n) T->cap <<= 1; T->key = malloc((size_t)T->cap * D); T->cnt = malloc(T->cap * sizeof(int)); T->used = malloc(T->cap); }
static int tab_add(Tab *T, const unsigned char *k) {
    uint64_t h = hkey(k) & (T->cap - 1);
    while (T->used[h] && memcmp(T->key + h * D, k, D)) h = (h + 1) & (T->cap - 1);
    if (!T->used[h]) { T->used[h] = 1; memcpy(T->key + h * D, k, D); T->cnt[h] = 0; }
    T->cnt[h]++; return (int)h;
}

// dimensión afín de φ(S): cuántas ecuaciones globales cumple el conjunto
static int HB[MAXD][MAXD], HR; static unsigned char H0[MAXD];
static int hull_dim(unsigned char *F, int n) {
    int (*B)[MAXD] = HB; int piv[MAXD], r = 0; memcpy(H0, F, D);
    for (int q = 1; q < n && r < D; q++) {
        int v[MAXD]; for (int d = 0; d < D; d++) v[d] = md(F[(size_t)q * D + d] - F[d]);
        for (int a = 0; a < r; a++) { int c = v[piv[a]]; if (c) for (int d = 0; d < D; d++) v[d] = md(v[d] - c * B[a][d]); }
        int f = -1; for (int d = 0; d < D; d++) if (v[d]) { f = d; break; }
        if (f < 0) continue;
        int s = inv[v[f]]; for (int d = 0; d < D; d++) v[d] = md(v[d] * s);
        memcpy(B[r], v, sizeof(int) * D); piv[r] = f; r++;
    }
    HR = r; return r;
}

// camino codicioso con residuos (eliminación incremental)
static void camino(unsigned char *F, int n, int base, int K, int *prof, Tab *T, unsigned char *W, char *in) {
    int nin = 0;
    for (int q = 0; q < n; q++) {
        int z = 1; for (int d = 0; d < D; d++) { W[(size_t)q * D + d] = md(F[(size_t)q * D + d] - F[(size_t)base * D + d]); z &= !W[(size_t)q * D + d]; }
        in[q] = z; nin += z;
    }
    prof[0] = nin;
    unsigned char key[MAXD];
    for (int k = 1; k <= K; k++) {
        memset(T->used, 0, T->cap);
        int best = -1, bestn = 0;
        for (int q = 0; q < n; q++) if (!in[q]) {
            unsigned char *w = W + (size_t)q * D; int f = 0; while (!w[f]) f++;
            int s = inv[w[f]]; for (int d = 0; d < D; d++) key[d] = w[d] * s % P;
            int h = tab_add(T, key), c = T->cnt[h];
            if (c > bestn) { bestn = c; best = h; }
        }
        // desempate uniforme entre todas las direcciones con el máximo (antes se
        // favorecían las últimas del array, que en el real son la misma órbita)
        if (bestn > 0) { int ties = 0; for (int h2 = 0; h2 < T->cap; h2++) if (T->used[h2] && T->cnt[h2] == bestn && rnd() % ++ties == 0) best = h2; }
        if (best < 0) { for (; k <= K; k++) prof[k] = nin; return; }
        unsigned char v[MAXD]; memcpy(v, T->key + (size_t)best * D, D);
        int f = 0; while (!v[f]) f++;
        for (int q = 0; q < n; q++) if (!in[q]) {
            unsigned char *w = W + (size_t)q * D; int c = w[f];
            if (c) { int z = 1; for (int d = 0; d < D; d++) { w[d] = md(w[d] - c * v[d]); z &= !w[d]; } if (z) { in[q] = 1; nin++; } }
        }
        prof[k] = nin;
    }
}

// Falso en la envoltura: n puntos al azar en el subespacio afín que ocupa φ(real).
static unsigned char *hull_null(int n) {
    unsigned char *F = malloc((size_t)n * D);
    for (int q = 0; q < n; q++) {
        int v[MAXD]; for (int d = 0; d < D; d++) v[d] = H0[d];
        for (int a = 0; a < HR; a++) { int c = rnd() % P; if (c) for (int d = 0; d < D; d++) v[d] += c * HB[a][d]; }
        for (int d = 0; d < D; d++) F[(size_t)q * D + d] = v[d] % P;
    }
    return F;
}
static void run(Set *S, unsigned char *Fin, int n, int ncam, int K, long *mx, double *mean, int *hd) {
    unsigned char *F = Fin, *W = malloc((size_t)n * D); char *in = malloc(n);
    if (!F) { F = malloc((size_t)n * D); for (int q = 0; q < n; q++) feat(S->v[q], F + (size_t)q * D); *hd = hull_dim(F, n); }
    else *hd = HR;
    for (int i = n - 1; i > 0; i--) {                  // barajar el orden
        int j = rnd() % (i + 1); unsigned char t[MAXD];
        memcpy(t, F + (size_t)i * D, D); memcpy(F + (size_t)i * D, F + (size_t)j * D, D); memcpy(F + (size_t)j * D, t, D);
    }
    Tab T; tab_init(&T, n); int prof[MAXD + 1];
    for (int k = 0; k <= K; k++) { mx[k] = 0; mean[k] = 0; }
    for (int c = 0; c < ncam; c++) {
        camino(F, n, rnd() % n, K, prof, &T, W, in);
        for (int k = 0; k <= K; k++) { mean[k] += prof[k]; if (prof[k] > mx[k]) mx[k] = prof[k]; }
    }
    for (int k = 0; k <= K; k++) mean[k] /= ncam;
    free(F); free(W); free(in); free(T.key); free(T.cnt); free(T.used);
}


// ---------- acción de H sobre los momentos ----------
static int binom[MAXP][MAXP], fidx[MAXP][MAXP], Mj0[MAXP];
static void build_action(void) {
    for (int n = 0; n < P; n++) { binom[n][0] = 1; for (int k = 1; k <= n; k++) binom[n][k] = (binom[n - 1][k - 1] + (k < n ? binom[n - 1][k] : 0)) % P; }
    for (int j = 0; j < P; j++) { long s = 0; for (int c = 0; c < P; c++) s += pw[c][j]; Mj0[j] = s % P; }
    memset(fidx, -1, sizeof fidx);
    for (int d = 0; d < D; d++) fidx[fj[d]][fl[d]] = d;
}
static int getM(const int *v, int j, int l) { return l == 0 ? Mj0[j] : v[fidx[j][l]]; }
static void act(const unsigned char *in, int t, int b, int u, int sg, unsigned char *out) {
    int v[MAXD], w[MAXD];
    for (int d = 0; d < D; d++) v[d] = in[d];
    for (int d = 0; d < D; d++) { int j = fj[d], l = fl[d]; long s = 0;        // columnas
        for (int m = 0; m <= j; m++) s += (long)binom[j][m] * pw[md(-t)][j - m] % P * getM(v, m, l);
        w[d] = s % P; }
    for (int d = 0; d < D; d++) { int j = fj[d], l = fl[d]; long s = 0;        // filas
        for (int m = 0; m <= l; m++) s += (long)binom[l][m] * pw[b][l - m] % P * getM(w, j, m);
        v[d] = s % P; }
    for (int d = 0; d < D; d++) { int e = pw[u][(fj[d] + fl[d]) % (P - 1)]; if (sg && (fl[d] & 1)) e = md(-e); out[d] = (long)v[d] * e % P; }
}
static void board_act(const unsigned char *x, int t, int b, int u, int sg, unsigned char *y) {
    unsigned char z[MAXP];
    for (int c = 0; c < P; c++) z[c] = md(x[(c + t) % P] + b);
    for (int c = 0; c < P; c++) { int v = md(u * z[c]); y[md(u * c)] = sg ? md(-v) : v; }
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "uso: ./grupos3 p clase lectura parámetro caminos K [semilla]\n"); return 1; }
    P = atoi(argv[1]); want = argv[2][0]; int par = atoi(argv[4]), ncam = atoi(argv[5]), K = atoi(argv[6]);
    if (argc > 7) rs ^= (uint64_t)atoll(argv[7]) * 0x9E3779B97F4A7C15ULL;
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_group(); build_reading(argv[3], par);
    if (ftype != 0) { fprintf(stderr, "solo lecturas sep/tot\n"); return 1; }
    build_action();
    xs[0] = 0; bt(1, 1u, 1u, 1u);
    int H = P * P * (P - 1) * 2;
    long nf = (long)REAL.n * P;                       // todas las filas b
    unsigned char *FR = malloc(nf * D);
    for (int q = 0; q < REAL.n; q++) for (int b = 0; b < P; b++) {
        unsigned char y[MAXP]; for (int c = 0; c < P; c++) y[c] = md(REAL.v[q][c] + b);
        feat(y, FR + ((size_t)q * P + b) * D);
    }
    // comprobación: la acción sobre momentos = momentos del tablero transformado
    int bad = 0;
    for (int it = 0; it < 200; it++) {
        int q = rnd() % REAL.n, t = rnd() % P, b = rnd() % P, u = 1 + rnd() % (P - 1), sg = rnd() & 1;
        unsigned char y[MAXP], f1[MAXD], f2[MAXD], f0[MAXD];
        feat(REAL.v[q], f0); act(f0, t, b, u, sg, f1); board_act(REAL.v[q], t, b, u, sg, y); feat(y, f2);
        bad += memcmp(f1, f2, D) != 0;
    }
    int orbits = nf / H;
    printf("p = %d  clase %c  lectura %s %d (D = %d)  soluciones %ld  |H| = %d  órbitas %d (resto %ld)  comprobación acción: %s\n",
           P, want, argv[3], par, D, nf, H, orbits, nf % H, bad ? "FALLA" : "ok");
    if (bad) return 1;
    int hr = hull_dim(FR, nf);
    // falso 1: uniforme en la envoltura; falso 2: órbitas por H de puntos al azar en la envoltura
    unsigned char *F1 = hull_null(nf);
    unsigned char *F2 = malloc(nf * D); long m = 0;
    unsigned char *seed = hull_null(orbits ? orbits : 1);
    for (int o = 0; o < orbits; o++)
        for (int t = 0; t < P; t++) for (int b = 0; b < P; b++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++)
            act(seed + (size_t)o * D, t, b, u, sg, F2 + (m++) * D);
    if (getenv("DEBUG")) {          // dimensión de órbitas parciales: real vs falso
        const char *nm[4] = {"filas b", "columnas t", "escala u", "signo"};
        for (int g = 0; g < 4; g++) {
            int lim = g == 3 ? 2 : P; int nreal = 0, nnull = 0;
            unsigned char *A = malloc((size_t)P * D), *B = malloc((size_t)P * D);
            int q = rnd() % REAL.n;
            for (int e = (g == 2); e < lim; e++) {
                int t = g == 1 ? e : 0, b = g == 0 ? e : 0, u = g == 2 ? e : 1, sg = g == 3 ? e : 0;
                unsigned char y[MAXP]; board_act(REAL.v[q], t, b, u, sg, y); feat(y, A + (size_t)nreal++ * D);
                act(seed, t, b, u, sg, B + (size_t)nnull++ * D);
            }
            printf("  DEBUG órbita por %s: dim real %d, dim falso %d (%d puntos)\n", nm[g], hull_dim(A, nreal), hull_dim(B, nnull), nreal);
            free(A); free(B);
        }
        hull_dim(FR, nf);
    }
    // falso 3: TABLEROS. Permutaciones al azar con φ dentro de la envoltura real
    // (cumplen todas las ecuaciones globales de las reales), y su órbita por H.
    int (*Bv)[MAXD] = HB; int piv[MAXD];
    for (int a = 0; a < HR; a++) { int f = 0; while (!Bv[a][f]) f++; piv[a] = f; }
    unsigned char *F3 = malloc(nf * D); long m3 = 0, tries = 0; int ok3 = 1;
    for (int o = 0; o < orbits && ok3; o++) {
        unsigned char x[MAXP], f[MAXD];
        for (;;) {
            if (++tries > 2000000000L) { ok3 = 0; break; }
            for (int i = 0; i < P; i++) x[i] = i;
            for (int i = P - 1; i > 0; i--) { int j = rnd() % (i + 1), t = x[i]; x[i] = x[j]; x[j] = t; }
            long m1s = 0, m2s = 0; for (int i = 0; i < P; i++) { m1s += (long)i * x[i]; m2s += (long)i * i % P * x[i]; }
            if (m1s % P || m2s % P) continue;
            feat(x, f);
            int v[MAXD]; for (int d = 0; d < D; d++) v[d] = md(f[d] - H0[d]);
            for (int a = 0; a < HR; a++) { int c = v[piv[a]]; if (c) for (int d = 0; d < D; d++) v[d] = md(v[d] - c * Bv[a][d]); }
            int z = 1; for (int d = 0; d < D; d++) z &= !v[d];
            if (z) break;
        }
        if (!ok3) break;
        for (int t = 0; t < P; t++) for (int b = 0; b < P; b++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++) {
            unsigned char y[MAXP]; board_act(x, t, b, u, sg, y); feat(y, F3 + (m3++) * D);
        }
    }
    printf("  falso de tableros: %s (%ld intentos)\n", ok3 ? "ok" : "NO SE ENCONTRARON", tries);
    if (getenv("DUMP")) {               // qué soluciones recoge un camino real
        int kd = atoi(getenv("DUMP")); Tab T; tab_init(&T, nf); int prof[MAXD + 1];
        unsigned char *W = malloc(nf * D); char *in = malloc(nf); int base = rnd() % nf;
        camino(FR, nf, base, kd, prof, &T, W, in);
        printf("  DUMP k=%d: %d soluciones (base %d)\n", kd, prof[kd], base);
        const unsigned char *xb = REAL.v[base / P]; int bb = base % P;
        for (long q = 0; q < nf; q++) if (in[q]) {
            const unsigned char *x = REAL.v[q / P]; int b = q % P;
            printf("   "); for (int c = 0; c < P; c++) printf(" %2d", md(x[c] + b));
            printf("   resta base:"); for (int c = 0; c < P; c++) printf(" %2d", md(x[c] + b - xb[c] - bb));
            printf("\n");
        }
        return 0;
    }
    if (K > hr) K = hr;
    long mr[MAXD + 1], m1[MAXD + 1], m2[MAXD + 1], mt[MAXD + 1]; double ar[MAXD + 1], a1[MAXD + 1], a2[MAXD + 1], at[MAXD + 1]; int h;
    run(0, FR, nf, ncam, K, mr, ar, &h);
    run(0, F1, nf, ncam, K, m1, a1, &h);
    run(0, F2, m, ncam, K, m2, a2, &h);
    if (ok3) run(0, F3, m3, ncam, K, mt, at, &h); else for (int k = 0; k <= K; k++) { mt[k] = 1; at[k] = 0; }
    printf("  dimensión de la envoltura %d\n", hr);
    printf("  k    real: máx   media | env. uniforme: máx  media | env. simetrías: máx  media | tableros simetrías: máx  media | real/unif real/sim real/tabl\n");
    for (int k = 0; k <= K; k++)
        printf("  %3d  %8ld %8.1f | %8ld %8.1f | %8ld %8.1f | %8ld %8.1f | %6.2f %6.2f %6.2f\n",
               k, mr[k], ar[k], m1[k], a1[k], m2[k], a2[k], mt[k], at[k], (double)mr[k] / m1[k], (double)mr[k] / m2[k], (double)mr[k] / mt[k]);
    return 0;
}
