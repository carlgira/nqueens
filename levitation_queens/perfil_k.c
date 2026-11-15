/* Primeros niveles: P_0..P_K para un tablero N x N (se corta en la profundidad K).
   Uso: ./perfil_k N K                                                               */
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
static int N, K; static u64 full; static u64 P[70];
static void rec(int k, u64 cols, u64 d1, u64 d2) {
    P[k]++;
    if (k == K) return;
    u64 free = full & ~(cols | d1 | d2);
    while (free) { u64 b = free & -free; free ^= b;
        rec(k + 1, cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1); }
}
int main(int argc, char **argv) {
    N = atoi(argv[1]); K = atoi(argv[2]); full = (N == 64) ? ~0ULL : ((1ULL << N) - 1);
    rec(0, 0, 0, 0);
    printf("%d", N);
    for (int k = 0; k <= K; k++) printf(" %llu", P[k]);
    printf("\n");
    return 0;
}
