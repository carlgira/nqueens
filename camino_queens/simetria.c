// simetria.c — ¿qué simetría tiene cada toroidal? (para saber qué camino le sirve)
//
// Simetrías: (i,x) -> s·g(i,x) + (t1,t2), con g uno de los 8 giros que permutan
// filas/columnas/diagonales, s ∈ Z_p^*, (t1,t2) traslación. Se descarta la
// identidad y las traslaciones puras.
//
// "Con centro" = alguna simetría con g diagonal (g = id o g = (i,-x)): en el
// marco del centro el polinomio x(i) solo usa exponentes k ≡ r (mod m) y lo
// recoge un camino de soporte (soporte.c). El resto se clasifica por el giro g
// de sus simetrías.
//
// Uso: ./simetria p

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 32
static int P, inv[MAXP];
static int md(int a) { a %= P; return a < 0 ? a + P : a; }

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
static const char *gname(int g) {
    static char buf[64]; const int *m = G[g];
    snprintf(buf, sizeof buf, "(%d,%d;%d,%d)", m[0], m[1], m[2], m[3]); return buf;
}

static int x[MAXP]; static long nsol;
static long cnt_cent, cnt_other, cnt_none, other_by_g[64], cent_and_other;
// ¿x invariante por s·G[g] + traslación?
static int has_sym(int g, int s) {
    int y[MAXP]; const int *m = G[g];
    for (int i = 0; i < P; i++) y[md(s * (m[0] * i + m[1] * x[i]))] = md(s * (m[2] * i + m[3] * x[i]));
    for (int t1 = 0; t1 < P; t1++) {
        if (g == 0 && s == 1 && t1 == 0) continue;         // traslaciones puras: y = x
        int t2 = md(x[0] - y[md(-t1)]), ok = 1;
        for (int i = 1; i < P && ok; i++) ok = x[i] == md(y[md(i - t1)] + t2);
        if (ok) { if (g == 0 && s == 1) { /* traslación pura no trivial: lineal */ } return 1; }
    }
    return 0;
}
static int is_diag(int g) { return G[g][1] == 0 && G[g][2] == 0; }
static void bt(int c, unsigned rows, unsigned dg, unsigned ad) {
    if (c == P) {
        nsol++;
        int cent = 0, other = 0, gs[64] = {0};
        for (int g = 0; g < nG; g++) for (int s = 1; s < P; s++) {
            if (g == 0 && s == 1) continue;
            if (has_sym(g, s)) { if (is_diag(g)) cent = 1; else { other = 1; gs[g] = 1; } }
        }
        if (cent) cnt_cent++; else if (other) cnt_other++; else cnt_none++;
        if (cent && other) cent_and_other++;
        if (!cent && other) for (int g = 0; g < nG; g++) other_by_g[g] += gs[g];
        return;
    }
    for (int r = 0; r < P; r++) {
        unsigned a = 1u << r, b = 1u << md(r - c), e = 1u << md(r + c);
        if ((rows & a) || (dg & b) || (ad & e)) continue;
        x[c] = r; bt(c + 1, rows | a, dg | b, ad | e);
    }
}

int main(int argc, char **argv) {
    P = atoi(argv[1]);
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_group();
    x[0] = 0; bt(1, 1u, 1u, 1u);
    long T = nsol * P;
    printf("p = %d   toroidales %ld\n", P, T);
    printf("  con centro (caminos de soporte):     %8ld  (%.1f %%)\n", cnt_cent * P, 100.0 * cnt_cent / nsol);
    printf("  sin centro pero con otra simetría:   %8ld  (%.1f %%)\n", cnt_other * P, 100.0 * cnt_other / nsol);
    printf("  sin ninguna simetría:                %8ld  (%.1f %%)\n", cnt_none * P, 100.0 * cnt_none / nsol);
    printf("  (con centro y además otra: %ld)\n", cent_and_other * P);
    printf("  giro g de las simetrías del grupo 'otra':");
    for (int g = 0; g < nG; g++) if (other_by_g[g]) printf("  %s:%ld", gname(g), other_by_g[g] * P);
    printf("\n");
    return 0;
}
