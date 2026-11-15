// grado.c — el camino de Newton para TODAS las toroidales de p primo.
//
// Cada tablero x se lee en la base de Newton: c_k = Δ^k x(0), k = 0..p-1,
// y se ordena con c_{p-1} como cifra más significativa. Un tablero de grado D
// (como polinomio i ↦ x_i mod p) tiene sus p-1-D cifras de arriba a 0: cae en
// el primer subárbol, de p^(D+1) tableros. El orden D2 de camino.c es el caso
// D = 1.
//
// Además, cada toroidal tiene 8 imágenes por los giros que permutan
// filas/columnas/diagonales; "grado mínimo" = el menor grado de las 8. Es el
// camino que mejor le va a esa solución.
//
// Uso:  ./grado p            enumera las toroidales (x_0 = 0, se multiplica por p)
//       ./grado p reps.txt   grados de las 8 imágenes de cada representante
//                            (una línea por solución: p enteros [y clase])

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 32
static int P, inv[MAXP];
static int md(int a) { a %= P; return a < 0 ? a + P : a; }

static int newton_degree(const int *x) {
    int d[MAXP]; memcpy(d, x, sizeof(int) * P);
    int deg = 0;
    for (int k = 1; k < P; k++) {                 // d pasa a ser Δ^k x
        int nz = 0;
        for (int i = 0; i + k < P; i++) { d[i] = md(d[i + 1] - d[i]); }
        if (d[0]) deg = k; (void)nz;
    }
    if (deg == 0 && x[0] == 0) { int any = 0; for (int i = 0; i < P; i++) any |= x[i]; if (!any) return -1; }
    return deg;
}

// 8 giros: clases de PGL2 que permutan las direcciones (1,0),(0,1),(1,1),(1,-1).
static int G[64][4], nG;
static void norm(int *m) { int f = m[0] ? m[0] : m[1]; int s = inv[f]; for (int k = 0; k < 4; k++) m[k] = md(m[k] * s); }
static void mul(const int *a, const int *b, int *c) {
    c[0] = md(a[0] * b[0] + a[1] * b[2]); c[1] = md(a[0] * b[1] + a[1] * b[3]);
    c[2] = md(a[2] * b[0] + a[3] * b[2]); c[3] = md(a[2] * b[1] + a[3] * b[3]);
}
static void build_group(void) {
    int gens[3][4] = {{0, 1, 1, 0}, {1, 0, 0, P - 1}, {1, 1, P - 1, 1}};   // (i,x)->(x,i), (i,-x), (i+x, x-i)
    nG = 1; G[0][0] = 1; G[0][1] = 0; G[0][2] = 0; G[0][3] = 1;
    for (int a = 0; a < nG; a++) for (int g = 0; g < 3; g++) {
        int c[4]; mul(gens[g], G[a], c); norm(c);
        int seen = 0; for (int b = 0; b < nG; b++) if (!memcmp(G[b], c, sizeof c)) seen = 1;
        if (!seen) memcpy(G[nG++], c, sizeof c);
    }
}
// Imagen de la solución por el giro m: puntos (i, x_i) -> m·(i, x_i); nueva función j -> y.
static int image(const int *x, const int *m, int *y) {
    for (int j = 0; j < P; j++) y[j] = -1;
    for (int i = 0; i < P; i++) {
        int j = md(m[0] * i + m[1] * x[i]), v = md(m[2] * i + m[3] * x[i]);
        if (y[j] >= 0) return 0;
        y[j] = v;
    }
    return 1;
}
static int min_degree(const int *x, int *degs) {
    int best = P;
    for (int g = 0; g < nG; g++) {
        int y[MAXP];
        if (!image(x, G[g], y)) { degs[g] = -2; continue; }
        degs[g] = newton_degree(y); if (degs[g] < best) best = degs[g];
    }
    return best;
}

// ---------- enumeración ----------
static int x[MAXP];
static long nsol, hist0[MAXP], histmin[MAXP];
static void bt(int c, unsigned rows, unsigned dg, unsigned ad) {
    if (c == P) {
        nsol++; hist0[newton_degree(x)]++;
        int degs[64]; histmin[min_degree(x, degs)]++;
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
    if (nG != 8) { fprintf(stderr, "grupo de %d elementos, se esperaban 8\n", nG); return 1; }

    if (argc > 2) {                                  // representantes
        FILE *f = fopen(argv[2], "r"); char line[1024];
        while (fgets(line, sizeof line, f)) {
            int v[MAXP + 1], n = 0; char *s = line, *e;
            for (;;) { long t = strtol(s, &e, 10); if (e == s) break; v[n++] = t; s = e; }
            if (n < P) continue;
            int degs[64]; int mn = min_degree(v, degs);
            printf("clase %d  grado %2d  mínimo %2d  las 8:", n > P ? v[P] : 0, newton_degree(v), mn);
            for (int g = 0; g < nG; g++) printf(" %d", degs[g]);
            printf("\n");
        }
        return 0;
    }

    x[0] = 0; bt(1, 1u, 1u, 1u);
    printf("p = %d   toroidales %ld  (enumeradas con x_0 = 0: %ld)\n", P, nsol * P, nsol);
    printf("  D   grado(x)   grado mínimo de las 8   subárbol p^(D+1)   densidad acumulada  (azar: %.3g)\n",
           (double)nsol * P / __builtin_powi((double)P, P));
    long c0 = 0, c1 = 0;
    for (int D = 0; D < P; D++) {
        if (!hist0[D] && !histmin[D]) continue;
        c0 += hist0[D] * P; c1 += histmin[D] * P;
        double sub = __builtin_powi((double)P, D + 1);
        printf("  %2d  %9ld   %9ld               %.3g          %.3g / %.3g\n", D, hist0[D] * P, histmin[D] * P, sub, c0 / sub, c1 / sub);
    }
    return 0;
}
