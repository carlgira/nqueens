/*
 * bloques.c — Soluciones de "bloques desplazados": la mitad de abajo es copia de la de arriba
 * desplazada s columnas (mod N):   S[r + N/2] = S[r] + s  (mod N),  r < N/2.
 * Escribe datos/tableros_N<N>_bloques.txt  y cuenta por desplazamiento s.
 * uso: ./bloques N
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static int N, H, s, c[64]; static long cnt[64], tot; static FILE *fo;
static int col[64], d1[128], d2[128];
static int ok_put(int r, int x) { return !col[x] && !d1[r + x] && !d2[r - x + N]; }
static void put(int r, int x, int v) { col[x] = v; d1[r + x] = v; d2[r - x + N] = v; }
static void bt(int r) {
    if (r == H) { for (int i = 0; i < N; i++) fprintf(fo, "%d%c", c[i], i < N - 1 ? ' ' : '\n'); cnt[s]++; tot++; return; }
    for (int x = 0; x < N; x++) {
        int y = (x + s) % N;
        if (!ok_put(r, x)) continue; put(r, x, 1);
        if (ok_put(r + H, y)) { put(r + H, y, 1); c[r] = x; c[r + H] = y; bt(r + 1); put(r + H, y, 0); }
        put(r, x, 0);
    }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); H = N / 2; mkdir("datos", 0755);
    char fn[64]; int sonly = argc > 2 ? atoi(argv[2]) : -1;
    if (sonly >= 0) snprintf(fn, sizeof fn, "datos/tableros_N%d_bloques_s%d.txt", N, sonly);
    else snprintf(fn, sizeof fn, "datos/tableros_N%d_bloques.txt", N);
    fo = fopen(fn, "w");
    if (sonly >= 0) { s = sonly; bt(0); }
    else for (s = 0; s < N; s++) bt(0);
    fclose(fo);
    printf("N=%d soluciones de bloques desplazados: %ld  por s:", N, tot);
    for (int k = 0; k < N; k++) if (cnt[k]) printf(" s=%d:%ld", k, cnt[k]);
    printf("\n");
    return 0;
}
