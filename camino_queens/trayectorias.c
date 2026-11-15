// trayectorias.c — caminos como trayectorias que saltan de solución en solución.
//
// Sin recorrer todos los tableros, pudiendo repetir y pudiendo pasar por
// soluciones que no cuentan. Una trayectoria es una regla corta b_0, b_1, …;
// se puntúan solo los pasos donde predice solución.
//
//   recta     b_k = s + k·v                       (salto constante; la fijan 2 soluciones)
//   parábola  b_k = s + k·v + C(k,2)·w            (salto que crece en w; la fijan 3)
//
// Solo cuentan los aciertos ADEMÁS de los que fijan la regla. Un acierto extra
// por azar tiene probabilidad ~ n / p^(p-3) (≈ 0), así que todo lo que salga es
// estructura; la pregunta es si es estructura de simetría.
//
// Falso: permutaciones al azar con Σ i·x ≡ Σ i²·x ≡ 0, en órbitas completas por
// H (columnas, filas, escala, signo), con el mismo número de órbitas.
// Como H actúa por transformaciones afines del tablero, lleva rectas en rectas
// y parábolas en parábolas: basta mirar una solución base por órbita de H.
// Se trabaja en el cociente por "subir filas": cada solución con x_0 = 0, así
// cada recta vertical (1,…,1), que es una simetría, cuenta como un punto. Se
// descartan las parábolas degeneradas (w paralelo a v: están sobre una recta).
//
// Uso: ./trayectorias p clase pares_por_base [semilla] [pares_falso]   (pares 0 = todos)
// Las bases son de la clase pedida; los aciertos cuentan en TODAS las toroidales.
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

// ---------- rectas por una base ----------
typedef struct { uint64_t h; int q, k; } Ent;
static int cmp_ent(const void *a, const void *b) { uint64_t x = ((const Ent *)a)->h, y = ((const Ent *)b)->h; return x < y ? -1 : x > y; }
static long line_hist[64], line_dir1[64];
static int best_line; static unsigned char best_line_dir[MAXP], best_line_base[MAXP]; static int best_line_lam[MAXP], best_line_nl;
static void lines(Big *S, int base, Ent *E) {
    int m = 0; const unsigned char *s = S->v[base];
    for (int q = 0; q < S->n; q++) if (q != base) {
        unsigned char d[MAXP]; int f = -1;
        for (int j = 0; j < P; j++) { d[j] = md(S->v[q][j] - s[j]); if (f < 0 && d[j]) f = j; }
        int lam = d[f], sc = inv[lam]; for (int j = 0; j < P; j++) d[j] = d[j] * sc % P;
        E[m].h = hrow(d); E[m].q = q; E[m].k = lam; m++;
    }
    qsort(E, m, sizeof(Ent), cmp_ent);
    for (int i = 0; i < m;) {
        int j = i; while (j < m && E[j].h == E[i].h) j++;
        int size = j - i + 1;
        if (size >= 3) {
            const unsigned char *q = S->v[E[i].q]; int dir1 = 1;
            for (int c = 1; c < P; c++) dir1 &= md(q[c] - s[c]) == md(q[0] - s[0]);
            if (dir1) line_dir1[size < 63 ? size : 63]++;
            else {
                line_hist[size < 63 ? size : 63]++;
                if (size > best_line) {
                    best_line = size; memcpy(best_line_base, s, P);
                    int f = -1; for (int c = 0; c < P; c++) { best_line_dir[c] = md(q[c] - s[c]); if (f < 0 && best_line_dir[c]) f = c; }
                    int sc = inv[best_line_dir[f]]; for (int c = 0; c < P; c++) best_line_dir[c] = best_line_dir[c] * sc % P;
                    best_line_nl = 0; best_line_lam[best_line_nl++] = 0;
                    for (int a = i; a < j; a++) best_line_lam[best_line_nl++] = E[a].k;
                }
            }
        }
        i = j;
    }
}

// ---------- parábolas por (base, segunda solución) ----------
static long par_hist[64]; static long par_pairs;
static int best_par; static int best_par_steps[MAXP], best_par_ns; static unsigned char best_par_s[MAXP], best_par_v[MAXP], best_par_w[MAXP];
// Para cada parábola con ≥4 aciertos: clases de las soluciones que toca y si
// algún elemento de H (≠ id) deja fijo el conjunto de aciertos.
static long rel_cnt[MAXP][MAXP][2], rel_none, share_hist[MAXP + 1], ray_lam[MAXP], ray_none;
static const char *PCLS; static long sig_cnt[64]; static char sig_name[64][MAXP + 1]; static int nsig; static long par_sym, par_tot;
static void classify_par(Big *S, int *idx, int nh) {
    char sg[MAXP + 1]; int m = 0;
    if (PCLS) { for (int a = 0; a < nh && m < MAXP; a++) sg[m++] = PCLS[idx[a]]; }
    else for (int a = 0; a < nh && m < MAXP; a++) sg[m++] = 'f';
    for (int a = 1; a < m; a++) { char t = sg[a]; int b = a - 1; while (b >= 0 && sg[b] > t) { sg[b + 1] = sg[b]; b--; } sg[b + 1] = t; }
    sg[m] = 0;
    int k; for (k = 0; k < nsig; k++) if (!strcmp(sig_name[k], sg)) break;
    if (k == nsig && nsig < 64) { strcpy(sig_name[nsig], sg); sig_cnt[nsig] = 0; nsig++; }
    if (k < 64) sig_cnt[k]++;
    int sym = 0; unsigned char y[MAXP];
    for (int t = 0; t < P && !sym; t++) for (int u = 1; u < P && !sym; u++) for (int g = 0; g < 2 && !sym; g++) {
        if (t == 0 && u == 1 && g == 0) continue;
        int ok = 1;
        for (int a = 0; a < nh && ok; a++) {
            board_act(S->v[idx[a]], t, 0, u, g, y); int r = big_find(S, y), in = 0;
            for (int b = 0; b < nh; b++) in |= r == idx[b];
            ok = in;
        }
        sym = ok;
    }
    par_tot++; par_sym += sym;
    // relación entre las asimétricas del conjunto de aciertos, y reinas compartidas con las de centro
    if (PCLS) {
        int ns[64], nn = 0, cs[64], nc = 0;
        for (int a = 0; a < nh; a++) { if (PCLS[idx[a]] == 'n') ns[nn++] = idx[a]; else if (PCLS[idx[a]] == 'c') cs[nc++] = idx[a]; }
        if (nn == 2) {
            int found = 0;
            for (int t = 0; t < P && !found; t++) for (int u = 1; u < P && !found; u++) for (int g = 0; g < 2 && !found; g++) {
                board_act(S->v[ns[0]], t, 0, u, g, y);
                if (!memcmp(y, S->v[ns[1]], P)) { found = 1; rel_cnt[t][u][g]++; }
            }
            if (!found) rel_none++;
        }
        if (nc == 2) {               // ¿c2 = λ·c1 (filas multiplicadas por λ)?
            int lam = -1;
            for (int l = 2; l < P && lam < 0; l++) { int ok = 1; for (int c = 0; c < P && ok; c++) ok = S->v[cs[1]][c] == md(l * S->v[cs[0]][c]); if (ok) lam = l; }
            if (lam > 0) ray_lam[lam]++; else ray_none++;
        }
        for (int a = 0; a < nn; a++) for (int b = 0; b < nc; b++) {
            int sh = 0; for (int c = 0; c < P; c++) sh += S->v[ns[a]][c] == S->v[cs[b]][c];
            // el cociente fija x_0 = 0; contar también con el mejor desplazamiento de filas
            int best = 0; for (int d = 0; d < P; d++) { int m2 = 0; for (int c = 0; c < P; c++) m2 += S->v[ns[a]][c] == md(S->v[cs[b]][c] + d); if (m2 > best) best = m2; }
            (void)sh; share_hist[best]++;
        }
    }
    if (getenv("DUMPPAR") && par_tot <= atoi(getenv("DUMPPAR"))) {
        printf("    parábola %ld:\n", par_tot);
        for (int a = 0; a < nh; a++) { printf("      %c", PCLS ? PCLS[idx[a]] : 'f'); for (int c = 0; c < P; c++) printf(" %2d", S->v[idx[a]][c]); printf("\n"); }
    }
}
static void parabolas(Big *S, int base, int tq, Ent *E) {
    const unsigned char *s = S->v[base], *t = S->v[tq]; int v[MAXP];
    for (int j = 0; j < P; j++) v[j] = md(t[j] - s[j]);
    long m = 0;
    for (int q = 0; q < S->n; q++) {
        if (q == base || q == tq) continue;
        const unsigned char *x = S->v[q];
        for (int k = 2; k < P; k++) {
            int ck = (k * (k - 1) / 2) % P; if (!ck) continue;
            int ic = inv[ck]; unsigned char w[MAXP]; int nz = 0;
            for (int j = 0; j < P; j++) { w[j] = md((x[j] - s[j] - k * v[j]) * ic); nz |= w[j]; }
            if (!nz) continue;                     // es la recta
            int par = 1, f = -1, r = 0;            // w ∥ v: parábola degenerada (sobre una recta)
            for (int jj = 0; jj < P; jj++) if (v[jj]) { f = jj; break; }
            if (f >= 0) { r = md(w[f] * inv[v[f]]); for (int jj = 0; jj < P && par; jj++) par = w[jj] == md(r * v[jj]); }
            if (par) continue;
            E[m].h = hrow(w); E[m].q = q; E[m].k = k; m++;
        }
    }
    qsort(E, m, sizeof(Ent), cmp_ent);
    par_pairs++;
    for (long i = 0; i < m;) {
        long j = i; while (j < m && E[j].h == E[i].h) j++;
        int hits = (int)(j - i) + 2;
        if (hits >= 4) {
            par_hist[hits < 63 ? hits : 63]++;
            { int idx[64], nh = 0; idx[nh++] = base; idx[nh++] = tq; for (long a = i; a < j && nh < 64; a++) idx[nh++] = E[a].q;
              if (getenv("DUMPPAR")) { printf("    pasos: 0 1"); for (long a = i; a < j; a++) printf(" %d", E[a].k); printf("\n"); }
              classify_par(S, idx, nh); }
            if (hits > best_par) {
                best_par = hits; best_par_ns = 0; best_par_steps[best_par_ns++] = 0; best_par_steps[best_par_ns++] = 1;
                for (long a = i; a < j; a++) best_par_steps[best_par_ns++] = E[a].k;
                memcpy(best_par_s, s, P); for (int c = 0; c < P; c++) best_par_v[c] = v[c];
                const unsigned char *x = S->v[E[i].q]; int k = E[i].k, ic = inv[(k * (k - 1) / 2) % P];
                for (int c = 0; c < P; c++) best_par_w[c] = md((x[c] - s[c] - k * v[c]) * ic);
            }
        }
        i = j;
    }
}

static void reset(void) {
    memset(line_hist, 0, sizeof line_hist); memset(line_dir1, 0, sizeof line_dir1); best_line = 0;
    memset(par_hist, 0, sizeof par_hist); par_pairs = 0; best_par = 0; nsig = 0; par_sym = par_tot = 0;
    memset(rel_cnt, 0, sizeof rel_cnt); rel_none = 0; memset(share_hist, 0, sizeof share_hist); memset(ray_lam, 0, sizeof ray_lam); ray_none = 0;
}
static void pr(const char *lab, const unsigned char *x) { printf("    %-6s", lab); for (int c = 0; c < P; c++) printf(" %2d", x[c]); printf("\n"); }
static void report(const char *name, int nbases) {
    printf("  [%s]  bases %d\n", name, nbases);
    printf("    otras rectas con ≥3 soluciones, por base:");
    int any = 0; for (int k = 3; k < 64; k++) if (line_hist[k]) { printf("  %d sol:%.2f", k, (double)line_hist[k] / nbases); any = 1; }
    if (!any) printf("  ninguna");
    printf("\n    parábolas con ≥4 soluciones, por par (base, segunda) [%ld pares]:", par_pairs);
    any = 0; for (int k = 4; k < 64; k++) if (par_hist[k]) { printf("  %d sol:%.4f", k, (double)par_hist[k] / par_pairs); any = 1; }
    if (!any) printf("  ninguna");
    printf("\n");
    if (par_tot) {
        printf("    clases de las soluciones que tocan (c centro, o giro, n sin simetría, f falso):");
        for (int k = 0; k < nsig; k++) printf("  %s:%ld", sig_name[k], sig_cnt[k]);
        printf("\n    con un elemento de H que deja fijos sus aciertos: %ld de %ld\n", par_sym, par_tot);
        if (PCLS) {
            printf("    relación entre sus 2 asimétricas (t columnas, u escala, signo):");
            for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int g = 0; g < 2; g++) if (rel_cnt[t][u][g]) printf("  (t=%d,u=%d,%s):%ld", t, u, g ? "−" : "+", rel_cnt[t][u][g]);
            printf("  ninguna:%ld\n    reinas compartidas asimétrica–centro (mejor fila):", rel_none);
            for (int k = 0; k <= P; k++) if (share_hist[k]) printf("  %d:%ld", k, share_hist[k]);
            printf("\n    sus 2 de centro cumplen c2 = λ·c1:");
            for (int l = 0; l < P; l++) if (ray_lam[l]) printf("  λ=%d:%ld", l, ray_lam[l]);
            printf("  no:%ld\n", ray_none);
        }
    }
    if (best_line) {
        printf("    mejor recta: %d soluciones, pasos λ =", best_line);
        for (int a = 0; a < best_line_nl; a++) printf(" %d", best_line_lam[a]); printf("\n");
        pr("base", best_line_base); pr("v", best_line_dir);
    }
    if (best_par) {
        printf("    mejor parábola: %d soluciones, pasos k =", best_par);
        for (int a = 0; a < best_par_ns; a++) printf(" %d", best_par_steps[a]); printf("\n");
        pr("s", best_par_s); pr("v", best_par_v); pr("w", best_par_w);
    }
}

static void analyze(const char *name, Big *S, int npairs, const char *cls) {
    PCLS = cls;
    // una base por órbita de H
    char *seen = calloc(S->n, 1); int *bases = malloc(S->n * sizeof(int)), nb = 0;
    for (int q = 0; q < S->n; q++) if (!seen[q]) {
        if (!cls || want == 't' || cls[q] == want) bases[nb++] = q; unsigned char y[MAXP];
        for (int t = 0; t < P; t++) for (int u = 1; u < P; u++) for (int sg = 0; sg < 2; sg++) {
            board_act(S->v[q], t, 0, u, sg, y); int r = big_find(S, y); if (r >= 0) seen[r] = 1;
        }
    }
    Ent *E = malloc((size_t)S->n * P * sizeof(Ent));
    reset();
    for (int i = 0; i < nb; i++) {
        lines(S, bases[i], E);
        if (npairs <= 0 || npairs >= S->n) { for (int t = 0; t < S->n; t++) if (t != bases[i]) parabolas(S, bases[i], t, E); }
        else for (int k = 0; k < npairs; k++) { int t = rnd() % S->n; if (t != bases[i]) parabolas(S, bases[i], t, E); }
    }
    report(name, nb);
    free(E); free(seen); free(bases);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "uso: ./trayectorias p clase pares_por_base [semilla]\n"); return 1; }
    P = atoi(argv[1]); want = argv[2][0]; int npairs = atoi(argv[3]);
    if (argc > 4) rs ^= (uint64_t)atoll(argv[4]) * 0x9E3779B97F4A7C15ULL;
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_group();
    xs[0] = 0; bt(1, 1u, 1u, 1u);
    long nf = REAL.n; int H = P * (P - 1) * 2;            // en el cociente; R = todas las toroidales
    Big R; big_init(&R, nf);
    for (int q = 0; q < REAL.n; q++) big_add(&R, REAL.v[q]);
    // falso: mismo número de órbitas por H
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
    printf("p = %d  clase %c  soluciones (x_0=0) %d  falso %d (%d órbitas, |H/filas| = %d)\n", P, want, R.n, Z.n, orbits, H);
    analyze("real (bases de la clase, aciertos en todas las toroidales)", &R, npairs, CLS);
    analyze("falso", &Z, argc > 5 ? atoi(argv[5]) : npairs, 0);
    return 0;
}
