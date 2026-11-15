/* Niveles por filas: P_k(N) = colocaciones sin ataques de k reinas en las k primeras filas.
   Uso: ./perfil_c N  -> imprime P_0 ... P_N                                              */
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
static int N; static unsigned full; static u64 P[40];
static void rec(int k, unsigned cols, unsigned d1, unsigned d2) {
    P[k]++;
    if (k == N) return;
    unsigned free = full & ~(cols | d1 | d2);
    while (free) { unsigned b = free & -free; free ^= b;
        rec(k + 1, cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1); }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); full = (1u << N) - 1;
    rec(0, 0, 0, 0);
    for (int k = 0; k <= N; k++) printf("%llu%c", P[k], k == N ? '\n' : ' ');
    return 0;
}
