/* Cuenta las soluciones de N reinas segun las columnas de las dos primeras filas.
   Uso: ./primeras N   -> imprime N lineas con N numeros: cuenta[c0][c1]            */
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
static int N; static unsigned full;
static u64 rec(unsigned cols, unsigned d1, unsigned d2) {
    if (cols == full) return 1;
    u64 n = 0; unsigned free = full & ~(cols | d1 | d2);
    while (free) { unsigned b = free & -free; free ^= b;
        n += rec(cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1); }
    return n;
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); full = (1u << N) - 1;
    for (int c0 = 0; c0 < N; c0++) {
        unsigned b0 = 1u << c0;
        unsigned cols = b0, d1 = (b0 << 1) & full, d2 = b0 >> 1;
        for (int c1 = 0; c1 < N; c1++) {
            unsigned b1 = 1u << c1; u64 n = 0;
            if (!((cols | d1 | d2) & b1))
                n = rec(cols | b1, ((d1 | b1) << 1) & full, (d2 | b1) >> 1);
            printf("%llu%c", n, c1 == N - 1 ? '\n' : ' ');
        }
    }
    return 0;
}
