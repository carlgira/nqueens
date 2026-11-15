/* Cuenta puntos de (Z_q)^N fuera de los hiperplanos x_i-x_j in {0, +-(i-j)}.
   = N reinas en N filas de un toro q x q sin atacarse.  Fija x_0 = 0 (divide por q). */
#include <stdio.h>
#include <stdlib.h>
typedef unsigned __int128 u128;
static int N, q; static u128 FULL;
static inline u128 rotl1(u128 m){ return ((m << 1) | (m >> (q-1))) & FULL; }
static inline u128 rotr1(u128 m){ return ((m >> 1) | ((m & 1) << (q-1))) & FULL; }
static inline int pc(u128 m){ return __builtin_popcountll((unsigned long long)m) + __builtin_popcountll((unsigned long long)(m>>64)); }
static long long rec(int r, u128 cols, u128 da, u128 db){
  /* da: columnas bloqueadas en la fila r por diagonales c-r ; db: por c+r */
  u128 free = ~(cols | da | db) & FULL;
  if (r == N-1) return pc(free);
  long long s = 0;
  while (free){ u128 bit = free & (~free + 1); free ^= bit;
    s += rec(r+1, cols|bit, rotl1(da|bit), rotr1(db|bit)); }
  return s;
}
int main(int argc, char **argv){
  N = atoi(argv[1]); q = atoi(argv[2]);
  FULL = (q == 128) ? ~(u128)0 : (((u128)1 << q) - 1);
  u128 b = 1;  /* x_0 = 0 */
  long long c = (N == 1) ? 1 : rec(1, b, rotl1(b), rotr1(b));
  printf("%lld\n", c);   /* = chi(q)/q */
  return 0;
}
