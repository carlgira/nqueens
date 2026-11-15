// zeta_huecos.c — de que estan hechos los huecos de la curva.  Por profundidad d:
//   posibles : d! parejas de cifras distintas, N (N-1) ... (N-d+1)
//   validos  : prefijos sin ataques entre sus d reinas
//   ocupados : prefijos validos que se completan a alguna solucion
// Un hueco es un prefijo posible que no esta ocupado: por ataque directo, o valido
// pero sin continuacion (callejon sin salida).
//
// cc -O2 -o zeta_huecos zeta_huecos.c && ./zeta_huecos 10 16
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 20
static int N;
static unsigned long long valid[MAXN + 1], occ[MAXN + 1];

static int rec(int r, unsigned cols, unsigned d1, unsigned d2) {
    valid[r]++;
    if (r == N) { occ[r]++; return 1; }
    unsigned avail = ~(cols | d1 | d2) & ((1u << N) - 1);
    int any = 0;
    while (avail) {
        unsigned b = avail & -avail;
        avail ^= b;
        any |= rec(r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1);
    }
    if (any) occ[r]++;
    return any;
}

int main(int argc, char **argv) {
    int n0 = atoi(argv[1]), n1 = atoi(argv[2]);
    printf("N,d,posibles,validos,ocupados\n");
    for (N = n0; N <= n1; N++) {
        memset(valid, 0, sizeof valid); memset(occ, 0, sizeof occ);
        rec(0, 0, 0, 0);
        double pos = 1;
        for (int d = 1; d <= N; d++) {
            pos *= N - d + 1;
            printf("%d,%d,%.0f,%llu,%llu\n", N, d, pos, valid[d], occ[d]);
        }
    }
    return 0;
}
