// soporte.c — caminos por grupos para las toroidales: base de monomios.
//
// Cada tablero se escribe como polinomio x(i) = Σ a_k i^k (mod p), k = 0..p-1
// (interpolación de Lagrange; es una biyección de los p^p tableros). El
// soporte de una solución es {k ≥ 1 : a_k ≠ 0} (a_0 solo sube/baja todo).
//
// Un camino = un conjunto U de exponentes. Se ordenan los tableros con los
// coeficientes fuera de U como cifras más significativas, y después a_0 y los
// de U. Las soluciones con soporte ⊆ U caen todas en el primer subárbol, de
// p^(|U|+1) tableros. Cada U recoge un grupo distinto.
//
// Uso: ./soporte p

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 32
static int P, inv[MAXP], Vinv[MAXP][MAXP];
static int md(int a) { a %= P; return a < 0 ? a + P : a; }

// a = Vinv · x, con V[i][k] = i^k.
static void build_vinv(void) {
    int A[MAXP][2 * MAXP];
    for (int i = 0; i < P; i++) {
        int pw = 1; for (int k = 0; k < P; k++) { A[i][k] = pw; pw = pw * i % P; }
        for (int k = 0; k < P; k++) A[i][P + k] = (i == k);
    }
    for (int c = 0; c < P; c++) {
        int r = c; while (!A[r][c]) r++;
        for (int k = 0; k < 2 * P; k++) { int t = A[r][k]; A[r][k] = A[c][k]; A[c][k] = t; }
        int s = inv[A[c][c]]; for (int k = 0; k < 2 * P; k++) A[c][k] = md(A[c][k] * s);
        for (int r2 = 0; r2 < P; r2++) if (r2 != c && A[r2][c]) {
            int f = A[r2][c]; for (int k = 0; k < 2 * P; k++) A[r2][k] = md(A[r2][k] - f * A[c][k]);
        }
    }
    for (int i = 0; i < P; i++) for (int k = 0; k < P; k++) Vinv[i][k] = A[i][P + k];
}
static unsigned support(const int *x) {
    unsigned m = 0;
    for (int k = 1; k < P; k++) { long s = 0; for (int i = 0; i < P; i++) s += (long)Vinv[k][i] * x[i]; if (s % P) m |= 1u << k; }
    return m;
}


// 8 giros que permutan filas/columnas/diagonales (clases de PGL2), como en grado.c.
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
static void image(const int *x, const int *m, int *y) {
    for (int i = 0; i < P; i++) y[md(m[0] * i + m[1] * x[i])] = md(m[2] * i + m[3] * x[i]);
}
static int use_rot;

// ---------- enumeración (x_0 = 0; cada una vale por p) ----------
typedef struct { unsigned m; long n; } Pat;
static Pat *pats; static int npat, cappat;
static int x[MAXP]; static long nsol;
// Mejor marco: desplazar columnas (i -> i + t) para poner el centro/cruce en 0.
// m = mayor divisor de p-1 con soporte ⊆ {k ≡ r mod m}: simetría x(ζi) = ζ^r x(i), ζ^m = 1.
static long hmin[MAXP + 1], hm[MAXP], hcent[8 * MAXP + 1];
static int best_m(unsigned mask) {
    for (int m = P - 1; m >= 2; m--) {
        if ((P - 1) % m) continue;
        int r = -1, ok = 1;
        for (int k = 1; k < P && ok; k++) if (mask >> k & 1) { if (r < 0) r = k % m; else if (k % m != r) ok = 0; }
        if (ok) return m;
    }
    return 1;
}
static void add(unsigned m) {
    for (int i = 0; i < npat; i++) if (pats[i].m == m) { pats[i].n++; return; }
    if (npat == cappat) { cappat = cappat ? 2 * cappat : 256; pats = realloc(pats, cappat * sizeof(Pat)); }
    pats[npat].m = m; pats[npat].n = 1; npat++;
}
static void bt(int c, unsigned rows, unsigned dg, unsigned ad) {
    if (c == P) {
        nsol++; add(support(x));
        int mn = P, bm = 1, ncent = 0, y[MAXP], z[MAXP];
        for (int g = 0; g < (use_rot ? nG : 1); g++)
        for (int t = 0; t < P; t++) {
            image(x, G[g], z);
            for (int i = 0; i < P; i++) y[i] = z[(i + t) % P];
            unsigned m = support(y); int s = __builtin_popcount(m), b = best_m(m);
            if (s < mn) mn = s; if (b > bm) bm = b; if (b >= 2) ncent++;
        }
        hmin[mn]++; hm[bm]++; hcent[ncent]++;
        return;
    }
    for (int r = 0; r < P; r++) {
        unsigned a = 1u << r, b = 1u << md(r - c), e = 1u << md(r + c);
        if ((rows & a) || (dg & b) || (ad & e)) continue;
        x[c] = r; bt(c + 1, rows | a, dg | b, ad | e);
    }
}
static void print_mask(unsigned m) {
    printf("{"); int first = 1;
    for (int k = 1; k < P; k++) if (m >> k & 1) { printf(first ? "%d" : ",%d", k); first = 0; }
    printf("}");
}
static int cmp_pat(const void *a, const void *b) {
    const Pat *x = a, *y = b; int ca = __builtin_popcount(x->m), cb = __builtin_popcount(y->m);
    if (ca != cb) return ca - cb; return (y->n > x->n) - (y->n < x->n);
}

int main(int argc, char **argv) {
    P = atoi(argv[1]);
    for (int a = 1; a < P; a++) for (int b = 1; b < P; b++) if (a * b % P == 1) inv[a] = b;
    build_vinv(); build_group(); use_rot = argc > 2;
    x[0] = 0; bt(1, 1u, 1u, 1u);
    qsort(pats, npat, sizeof(Pat), cmp_pat);
    long tot = nsol * P;
    printf("p = %d   toroidales %ld   soportes distintos %d\n", P, tot, npat);

    // histograma por tamaño de soporte
    printf("  tamaño de soporte → soluciones:");
    for (int s = 0; s <= P; s++) { long c = 0; for (int i = 0; i < npat; i++) if (__builtin_popcount(pats[i].m) == s) c += pats[i].n; if (c) printf("  %d:%ld", s, c * P); }
    printf("\n");
    printf("  con el mejor desplazamiento de columna:\n    soporte mínimo → soluciones:");
    for (int k = 0; k <= P; k++) if (hmin[k]) printf("  %d:%ld", k, hmin[k] * P);
    printf("\n    simetría x(ζi)=ζ^r x(i), orden m de ζ → soluciones:");
    for (int k = 1; k < P; k++) if (hm[k]) printf("  m=%d:%ld", k, hm[k] * P);
    printf("\n    nº de marcos (giro, columna) con centro → soluciones:");
    for (int k = 0; k <= 8 * P; k++) if (hcent[k]) printf("  %d:%ld", k, hcent[k] * P);
    printf("\n\n");

    // caminos: para cada soporte U que aparece, cuántas caben en su subárbol
    printf("  camino U (exponentes libres)   en el subárbol   tamaño p^(|U|+1)   densidad   (propias)\n");
    int shown = 0;
    for (int i = 0; i < npat && shown < 40; i++) {
        unsigned U = pats[i].m; long in = 0;
        for (int j = 0; j < npat; j++) if ((pats[j].m & ~U) == 0) in += pats[j].n;
        double sub = __builtin_powi((double)P, __builtin_popcount(U) + 1);
        printf("  "); print_mask(U);
        printf("%*s %10ld   %14.4g   %9.3g   (%ld)\n", (int)(30 - 3 * __builtin_popcount(U)) > 0 ? (int)(30 - 3 * __builtin_popcount(U)) : 1, "",
               in * P, sub, in * P / sub, pats[i].n * P);
        shown++;
    }
    if (npat > shown) printf("  … %d soportes más\n", npat - shown);
    return 0;
}
