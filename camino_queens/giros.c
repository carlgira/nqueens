// giros.c — ¿por qué girar las filas (r -> r+c mod N) encadena soluciones normales?
//
// Teoría. Al girar c, un par de reinas (columnas i<j, d = j-i, filas a<b) solo
// cambia su diferencia si exactamente una da la vuelta, y entonces chocan si
// b - a = N - d: es un par que se ataca en el toro pero no en el tablero
// ("par de costura"). Ese par prohíbe los giros c ∈ [N-b, N-a), N-d seguidos.
// W(s) = nº de pares de costura; W = 0 ⇔ s es toroidal.
// La trayectoria s, s+1, s+2, … vale hasta el primer giro prohibido.
//
// Enumera todas las soluciones de N (sin guardarlas) y para cada no toroidal:
//   L    = soluciones en la trayectoria más larga s, s+c, s+2c, … (cualquier c)
//   L1   = la de paso c = 1, medida y predicha por las ventanas (deben coincidir)
//   W, y las distancias d de sus pares de costura
//
// Uso: ./giros N [umbral_para_ejemplos]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 20
static int N, TH;
static int x[MAXN];
static long nsol, ntor, histL[MAXN + 1], histW[64], histWlong[64], histL1[MAXN + 1], mism, nlong, dlong[MAXN + 1], dall[MAXN + 1];
static long histWL[16][MAXN + 1];      // W × L
static int shown; static long common_long, common_all;
static int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

static int valid_rot(int c) {                 // ¿x + c (mod N) es solución normal?
    unsigned long d1 = 0, d2 = 0;
    for (int i = 0; i < N; i++) {
        int r = (x[i] + c) % N; unsigned long b1 = 1UL << (r + i), b2 = 1UL << (r - i + N);
        if ((d1 & b1) || (d2 & b2)) return 0;
        d1 |= b1; d2 |= b2;
    }
    return 1;
}

static void leaf(void) {
    nsol++;
    // pares de costura
    int W = 0, bad[MAXN] = {0}, ds[64], nd = 0, inv[MAXN] = {0};
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        int d = j - i, a = x[i] < x[j] ? x[i] : x[j], b = x[i] < x[j] ? x[j] : x[i];
        if (b - a == N - d) { W++; inv[i]++; inv[j]++; if (nd < 64) ds[nd++] = d; for (int c = N - b; c < N - a; c++) bad[c] = 1; }
    }
    int common = 0; for (int i = 0; i < N; i++) if (W && inv[i] == W) common = 1;   // una reina en todos los pares
    if (W == 0) { ntor++; return; }
    int ok[MAXN]; ok[0] = 1;
    for (int c = 1; c < N; c++) ok[c] = valid_rot(c);
    for (int c = 1; c < N; c++) if (ok[c] != !bad[c]) mism++;          // teoría vs medida
    int L = 1;
    for (int c = 1; c < N; c++) { int per = N / gcd(c, N), l = 1; while (l < per && ok[(l * c) % N]) l++; if (l > L) L = l; }
    int L1 = 1; while (L1 < N && ok[L1]) L1++;
    if (common) common_all++;
    histL[L]++; histL1[L1]++; histW[W < 63 ? W : 63]++; histWL[W < 15 ? W : 15][L]++;
    for (int k = 0; k < nd; k++) dall[ds[k]]++;
    if (L >= TH) {
        nlong++; histWlong[W < 63 ? W : 63]++; common_long += common;
        for (int k = 0; k < nd; k++) dlong[ds[k]]++;
        if (shown < 4) {
            shown++;
            printf("  ejemplo L=%d W=%d:", L, W); for (int i = 0; i < N; i++) printf(" %d", x[i]);
            printf("   pares de costura (i,j | filas):");
            for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { int a = x[i], b = x[j]; if ((b > a ? b - a : a - b) == N - (j - i)) printf(" (%d,%d|%d,%d)", i, j, a, b); }
            printf("\n");
        }
    }
}
static void bt(int c, unsigned rows, unsigned long d1, unsigned long d2) {
    if (c == N) { leaf(); return; }
    for (int r = 0; r < N; r++) {
        unsigned a = 1u << r; unsigned long b = 1UL << (r + c), e = 1UL << (r - c + N);
        if ((rows & a) || (d1 & b) || (d2 & e)) continue;
        x[c] = r; bt(c + 1, rows | a, d1 | b, d2 | e);
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]); TH = argc > 2 ? atoi(argv[2]) : N;
    bt(0, 0, 0, 0);
    long nn = nsol - ntor;
    printf("N = %d   soluciones %ld   toroidales %ld   no toroidales %ld   teoría≠medida: %ld\n", N, nsol, ntor, nn, mism);
    printf("  trayectoria más larga L (soluciones):"); for (int k = 1; k <= N; k++) if (histL[k]) printf("  %d:%ld", k, histL[k]); printf("\n");
    printf("  con paso c=1 (L1):                  "); for (int k = 1; k <= N; k++) if (histL1[k]) printf("  %d:%ld", k, histL1[k]); printf("\n");
    printf("  pares de costura W:                 "); for (int k = 1; k < 64; k++) if (histW[k]) printf("  %d:%ld", k, histW[k]); printf("\n");
    printf("  L media según W:"); for (int w = 1; w < 16; w++) { long n = 0, s = 0; for (int k = 1; k <= N; k++) { n += histWL[w][k]; s += histWL[w][k] * k; } if (n) printf("  W=%d: %.2f (%ld)", w, (double)s / n, n); } printf("\n");
    printf("  distancia d de los pares de costura, todas: "); for (int d = 1; d < N; d++) if (dall[d]) printf(" %d:%ld", d, dall[d]); printf("\n");
    if (nlong) {
        printf("  con L ≥ %d (%ld soluciones): W →", TH, nlong); for (int k = 1; k < 64; k++) if (histWlong[k]) printf("  %d:%ld", k, histWlong[k]); printf("\n");
        printf("     distancia d de sus pares de costura: "); for (int d = 1; d < N; d++) if (dlong[d]) printf(" %d:%ld", d, dlong[d]); printf("\n");
        printf("     con una misma reina en todos sus pares de costura: %ld de %ld   (en todas las no toroidales: %ld de %ld)\n", common_long, nlong, common_all, nn);
    }
    return 0;
}
