// zeta_fractal.c — escribe las soluciones como lineas de N cifras hexadecimales (0..N-1),
// en orden numerico, para dibujar el fractal de prefijos con plot_fractal.py.
//
// cc -O2 -o zeta_fractal zeta_fractal.c && ./zeta_fractal 14 > sols_14.txt
#include <stdio.h>
#include <stdlib.h>

static int N, cur[16];

static void rec(int r, unsigned cols, unsigned d1, unsigned d2) {
    if (r == N) {
        for (int i = 0; i < N; i++) putchar("0123456789abcdef"[cur[i]]);
        putchar('\n');
        return;
    }
    unsigned avail = ~(cols | d1 | d2) & ((1u << N) - 1);
    while (avail) {
        unsigned b = avail & -avail;
        avail ^= b;
        cur[r] = __builtin_ctz(b);
        rec(r + 1, cols | b, (d1 | b) << 1, (d2 | b) >> 1);
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    rec(0, 0, 0, 0);
    return 0;
}
