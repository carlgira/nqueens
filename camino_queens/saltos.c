// saltos.c — trayectorias geométricas b_{k+1} = M·b_k + c que encadenan soluciones.
//
// Un paso de M encadena dos soluciones; "cada d pasos" es lo mismo con M^d, así
// que basta buscar cadenas s_0 -> s_1 -> s_2 -> … con un mismo G(b) = M·b + c.
// Con M general cualquier cadena corta se ajusta, así que se usan familias
// pequeñas de M (pocos parámetros), y solo cuentan las soluciones que la cadena
// acierta además de las que fijan G:
//
//   diagonal     G(x)_i = λ_i·x_i + c_i   (el salto de cada columna se
//                multiplica por λ_i: salto geométrico columna a columna).
//                La fijan 3 soluciones; cuenta desde la 4.ª.
//                Si todos los λ_i son 1 es una recta (ya vista); si todos son
//                iguales, G = λ·x + c (escalar las filas).
//   permutación  G(x) = x∘π (M = matriz de permutación de columnas).
//                La fijan 2 soluciones; cuenta desde la 3.ª.
//                π traslación (i -> i+τ) es una simetría y se excluye;
//                π afín (i -> u·i + τ) se separa de π no afín.
//
// Cociente por subir filas (x_0 = 0); una base por órbita de H (que conjuga
// cada familia en sí misma); aciertos en TODAS las toroidales; falso como en
// trayectorias.c.
//
// Uso: ./saltos p clase segundas_por_base [semilla]   (0 = todas)
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
static Set REAL; static char want; static char *CLS;   // todas las toroidales y su clase
static int xs[MAXP];
static void bt(int c, unsigned rows, unsigned dg, unsigned ad) {
    if (c == P) { if (REAL.n % 4096 == 0) CLS = realloc(CLS, REAL.n + 4096); CLS[REAL.n] = classify(xs); push(&REAL, xs); return; }
    for (int r = 0; r < P; r++) {
        unsigned a = 1u << r, b = 1u << md(r - c), e = 1u << md(r + c);
        if ((rows & a) || (dg & b) || (ad & e)) continue;
        xs[c] = r; bt(c + 1, rows | a, dg | b, ad | e);
    }
}

// ---------- conjunto completo con búsqueda por hash ----------
static uint64_t hrow(const unsigned char *v) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < P; i++) h = (h ^ v[i]) * 1099511628211ULL; return h ^ (h >> 29); }
typedef struct { int n, cap, hcap; row *v; int *slot; } Big;
static int big_find(Big *S, const unsigned char *x) {
    uint64_t k = hrow(x) & (S->hcap - 1);
    while (S->slot[k] >= 0) { if (!memcmp(S->v[S->slot[k]], x, P)) return S->slot[k]; k = (k + 1) & (S->hcap - 1); }
    return -1;
}
static void big_init(Big *S, int cap) {
    S->n = 0; S->cap = cap; S->v = malloc((size_t)cap * sizeof(row));
    S->hcap = 1; while (S->hcap < 2 * cap) S->hcap <<= 1;
    S->slot = malloc(S->hcap * sizeof(int)); memset(S->slot, -1, S->hcap * sizeof(int));
}
static int big_add(Big *S, const unsigned char *x) {
    uint64_t k = hrow(x) & (S->hcap - 1);
    while (S->slot[k] >= 0) { if (!memcmp(S->v[S->slot[k]], x, P)) return 0; k = (k + 1) & (S->hcap - 1); }
    if (S->n == S->cap) { fprintf(stderr, "Big lleno\n"); exit(1); }
    memset(S->v[S->n], 0, sizeof(row)); memcpy(S->v[S->n], x, P); S->slot[k] = S->n++; return 1;
}
static void board_act(const unsigned char *x, int t, int b, int u, int sg, unsigned char *y) {
    unsigned char z[MAXP];
    for (int c = 0; c < P; c++) z[c] = md(x[(c + t) % P] + b);
    for (int c = 0; c < P; c++) { int v = md(u * z[c]); y[md(u * c)] = sg ? md(-v) : v; }
    int c0 = y[0]; for (int c = 0; c < P; c++) y[c] = md(y[c] - c0);      // cociente: x_0 = 0
}
// añade la órbita completa por H; devuelve cuántos nuevos
static int add_orbit(Big *S, const unsigned char *x) {
    int nw = 0; unsigned char y[MAXP];
    for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++) {
        board_act(x, t, 0, u, sg, y); nw += big_add(S, y);
    }
    return nw;
}


// ---------- estadística de cadenas ----------
typedef struct {
    long hist[3][64];                  // tipo × nº de soluciones distintas en la cadena
    long sig_cnt[128]; char sig[128][12]; int nsig;
    int best[3]; unsigned char ex[3][8][MAXP]; int exn[3]; int exlam[3][MAXP];
    long expl[3], unexpl[3];
} Stat;
static const char *TYPE_DIAG[3] = {"λ distintos", "λ·x + c (escalar filas)", "?"};
static const char *TYPE_PERM[3] = {"π no afín", "π afín no traslación", "?"};
static const char *CL;
// ¿la cadena se explica por simetría? último ∈ H·primero, o ∈ H·transpuesta(primero)

static int in_orbit(const unsigned char *a, const unsigned char *b) {
    unsigned char y[MAXP];
    for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int g = 0; g < 2; g++) { board_act(a, t, 0, u, g, y); if (!memcmp(y, b, P)) return 1; }
    return 0;
}
// imagen por uno de los 8 giros G[g] (puntos (i, x_i) -> G·(i, x_i)), en el cociente
static void giro(const unsigned char *x, int g, unsigned char *y) {
    const int *m = G[g];
    for (int i = 0; i < P; i++) y[md(m[0] * i + m[1] * x[i])] = md(m[2] * i + m[3] * x[i]);
    int c0 = y[0]; for (int i = 0; i < P; i++) y[i] = md(y[i] - c0);
}
static void record(Stat *st, int type, unsigned char (*ch)[MAXP], int *idx, int L, const int *lam) {
    st->hist[type][L < 63 ? L : 63]++;
    {   int e = 0;
        for (int a0 = 0; a0 < L && !e; a0++) for (int g = 0; g < nG && !e; g++) {
            unsigned char tr[MAXP]; giro(ch[a0], g, tr);
            for (int a = a0 + 1; a < L && !e; a++) e = in_orbit(tr, ch[a]);
        }
        if (e) st->expl[type]++; else st->unexpl[type]++; }
    if (CL) {
        char sg[12]; int m = 0; for (int a = 0; a < L && m < 11; a++) sg[m++] = CL[idx[a]];
        for (int a = 1; a < m; a++) { char t = sg[a]; int b = a - 1; while (b >= 0 && sg[b] > t) { sg[b + 1] = sg[b]; b--; } sg[b + 1] = t; }
        sg[m] = 0; char key[12]; snprintf(key, sizeof key, "%d%s", type, sg);
        int k; for (k = 0; k < st->nsig; k++) if (!strcmp(st->sig[k], key)) break;
        if (k == st->nsig && st->nsig < 128) { strcpy(st->sig[k], key); st->sig_cnt[k] = 0; st->nsig++; }
        if (k < 128) st->sig_cnt[k]++;
    }
    if (L > st->best[type]) {
        st->best[type] = L; st->exn[type] = L < 8 ? L : 8;
        for (int a = 0; a < st->exn[type]; a++) memcpy(st->ex[type][a], ch[a], MAXP);
        if (lam) memcpy(st->exlam[type], lam, sizeof(int) * MAXP);
    }
}
static void print_stat(Stat *st, const char **names, int fit) {
    printf("      algún par de la cadena está relacionado por el grupo completo (H y los 8 giros): %s %ld sí / %ld no;  %s %ld sí / %ld no\n",
           names[0], st->expl[0], st->unexpl[0], names[1], st->expl[1], st->unexpl[1]);
    for (int ty = 0; ty < 2; ty++) {
        printf("      %-26s", names[ty]); int any = 0;
        for (int L = fit + 1; L < 64; L++) if (st->hist[ty][L]) { printf("  %d sol:%ld", L, st->hist[ty][L]); any = 1; }
        if (!any) printf("  ninguna con %d o más", fit + 1);
        printf("\n");
    }
    if (st->nsig) {
        printf("      clases de las cadenas (tipo+clases):");
        for (int k = 0; k < st->nsig; k++) printf("  %s:%ld", st->sig[k], st->sig_cnt[k]);
        printf("\n");
    }
    for (int ty = 0; ty < 2; ty++) if (st->best[ty] > fit) {
        printf("      mejor cadena (%s): %d soluciones\n", names[ty], st->best[ty]);
        for (int a = 0; a < st->exn[ty]; a++) { printf("       "); for (int c = 0; c < P; c++) printf(" %2d", st->ex[ty][a][c]); printf("\n"); }
        if (names == TYPE_DIAG) { printf("       λ:"); for (int c = 0; c < P; c++) printf(" %2d", st->exlam[ty][c]); printf("\n"); }
    }
}

// ---------- cadenas diagonales: (s, t, u) fijan λ_i, c_i ----------
static void diag_chains(Big *S, int sb, int tb, Stat *st) {
    const unsigned char *s = S->v[sb], *t = S->v[tb];
    int d1[MAXP], id1[MAXP];
    for (int i = 0; i < P; i++) { d1[i] = md(t[i] - s[i]); id1[i] = d1[i] ? inv[d1[i]] : 0; }
    for (int ub = 0; ub < S->n; ub++) {
        if (ub == sb || ub == tb) continue;
        const unsigned char *u = S->v[ub]; int lam[MAXP], cc[MAXP], ok = 1, all1 = 1, same = 1, l0 = -1;
        for (int i = 0; i < P && ok; i++) {
            int d2 = md(u[i] - t[i]);
            if (!d1[i]) { if (d2) ok = 0; lam[i] = 1; cc[i] = 0; continue; }
            lam[i] = d2 * id1[i] % P; cc[i] = md(t[i] - lam[i] * s[i]);
            if (lam[i] != 1) all1 = 0;
            if (l0 < 0) l0 = lam[i]; else if (lam[i] != l0) same = 0;
        }
        if (!ok || all1) continue;                      // imposible, o es una recta
        // ¿predice la 4.ª?
        unsigned char ch[64][MAXP]; int idx[64], L = 3;
        memcpy(ch[0], s, MAXP); memcpy(ch[1], t, MAXP); memcpy(ch[2], u, MAXP); idx[0] = sb; idx[1] = tb; idx[2] = ub;
        for (;;) {
            unsigned char v[MAXP]; memset(v, 0, MAXP);
            for (int i = 0; i < P; i++) v[i] = md(lam[i] * ch[L - 1][i] + cc[i]);
            int r = big_find(S, v); if (r < 0) break;
            int rep = 0; for (int a = 0; a < L; a++) rep |= idx[a] == r;
            if (rep || L >= 63) break;
            memcpy(ch[L], v, MAXP); idx[L] = r; L++;
        }
        if (L >= 4) record(st, same ? 1 : 0, ch, idx, L, lam);
    }
}

// ---------- cadenas por permutación de columnas: (s, t+c) fijan π ----------
static void perm_chains(Big *S, int sb, int tb, Stat *st) {
    const unsigned char *s = S->v[sb], *t = S->v[tb]; int sinv[MAXP];
    for (int i = 0; i < P; i++) sinv[s[i]] = i;
    for (int c = 0; c < P; c++) {
        int pi[MAXP]; for (int i = 0; i < P; i++) pi[i] = sinv[md(t[i] + c)];
        // tipo de π
        int u = md(pi[1] - pi[0]), tau = pi[0], aff = 1;
        for (int i = 0; i < P && aff; i++) aff = pi[i] == md(u * i + tau);
        if (aff && u == 1) continue;                    // traslación de columnas: simetría
        int type = aff ? 1 : 0;
        unsigned char ch[64][MAXP]; int idx[64], L = 2;
        memcpy(ch[0], s, MAXP); memcpy(ch[1], t, MAXP); idx[0] = sb; idx[1] = tb;
        for (;;) {
            unsigned char v[MAXP]; memset(v, 0, MAXP);
            for (int i = 0; i < P; i++) v[i] = ch[L - 1][pi[i]];
            int c0 = v[0]; for (int i = 0; i < P; i++) v[i] = md(v[i] - c0);
            int r = big_find(S, v); if (r < 0) break;
            int rep = 0; for (int a = 0; a < L; a++) rep |= idx[a] == r;
            if (rep || L >= 63) break;
            memcpy(ch[L], v, MAXP); idx[L] = r; L++;
        }
        if (L >= 3) record(st, type, ch, idx, L, 0);
    }
}

static void analyze(const char *name, Big *S, int nsec, const char *cls) {
    CL = cls;
    char *seen = calloc(S->n, 1); int *bases = malloc(S->n * sizeof(int)), nb = 0;
    for (int q = 0; q < S->n; q++) if (!seen[q]) {
        if (!cls || want == 't' || cls[q] == want) bases[nb++] = q;
        unsigned char y[MAXP];
        for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++) {
            board_act(S->v[q], t, 0, u, sg, y); int r = big_find(S, y); if (r >= 0) seen[r] = 1;
        }
    }
    Stat *sd = calloc(1, sizeof(Stat)), *sp = calloc(1, sizeof(Stat)); long npairs = 0;
    for (int b = 0; b < nb; b++) {
        int all = nsec <= 0 || nsec >= S->n;
        for (int t = 0; t < S->n; t++) if (t != bases[b]) perm_chains(S, bases[b], t, sp);   // barata: siempre todas
        for (int k = 0; k < (all ? S->n : nsec); k++) {
            int t = all ? k : (int)(rnd() % S->n); if (t == bases[b]) continue;
            diag_chains(S, bases[b], t, sd);
            npairs++;
        }
    }
    printf("  [%s]  bases %d, pares para la diagonal %ld (la permutación usa todos)\n", name, nb, npairs);
    printf("    diagonal (la fijan 3; se cuentan cadenas con ≥4):\n"); print_stat(sd, TYPE_DIAG, 3);
    printf("    permutación de columnas (la fijan 2; cadenas con ≥3):\n"); print_stat(sp, TYPE_PERM, 2);
    free(sd); free(sp); free(seen); free(bases);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: ./saltos p clase segundas_por_base [semilla]\n"); return 1; }
    P = atoi(argv[1]); want = argv[2][0]; int nsec = atoi(argv[3]);
    if (argc > 4) rs ^= (uint64_t)atoll(argv[4]) * 0x9E3779B97F4A7C15ULL;
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_group();
    xs[0] = 0; bt(1, 1u, 1u, 1u);
    long nf = REAL.n; int H = P * (P - 1) * 2;
    Big R; big_init(&R, nf);
    for (int q = 0; q < REAL.n; q++) big_add(&R, REAL.v[q]);
    int orbits = (int)((nf + H - 1) / H);
    Big Z; big_init(&Z, (long)orbits * H + H);
    for (int o = 0; o < orbits;) {
        unsigned char x[MAXP];
        for (int i = 0; i < P; i++) x[i] = i;
        for (int i = P - 1; i > 0; i--) { int j = rnd() % (i + 1), t = x[i]; x[i] = x[j]; x[j] = t; }
        long m1 = 0, m2 = 0; for (int i = 0; i < P; i++) { m1 += (long)i * x[i]; m2 += (long)i * i % P * x[i]; }
        if (m1 % P || m2 % P) continue;
        { int c0 = x[0]; for (int i = 0; i < P; i++) x[i] = md(x[i] - c0); }
        if (big_find(&Z, x) >= 0) continue;
        add_orbit(&Z, x); o++;
    }
    printf("p = %d  clase %c  toroidales (x_0=0) %d  falso %d\n", P, want, R.n, Z.n);
    analyze("real (bases de la clase, aciertos en todas las toroidales)", &R, nsec, CLS);
    analyze("falso", &Z, nsec, 0);
    return 0;
}
