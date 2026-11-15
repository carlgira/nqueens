// Cuenta caminos hamiltonianos (dirigidos: cada orden de visita cuenta) en un grafo pequeño con multiplicidades.
// Entrada: n, luego líneas "i j m" (m = nº de aristas distintas de i a j). Salida: nº de caminos como
// secuencias de clases (simple) y ponderado por multiplicidad (secuencias de clases + etiqueta T en cada paso).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned long long u64;
typedef unsigned __int128 u128;
int n, adj[32][32];
u64 full, simple_cnt = 0;
u128 weighted = 0;
void dfs(int v, u64 mask, int depth, u128 w) {
    if (mask == full) { simple_cnt++; weighted += w; return; }
    for (int u = 0; u < n; u++)
        if (adj[v][u] && !(mask >> u & 1)) dfs(u, mask | (1ULL << u), depth + 1, w * adj[v][u]);
}
int main() {
    int i, j, m;
    scanf("%d", &n);
    full = (n == 64) ? ~0ULL : ((1ULL << n) - 1);
    while (scanf("%d %d %d", &i, &j, &m) == 3) adj[i][j] = m;
    for (int s = 0; s < n; s++) dfs(s, 1ULL << s, 1, 1);
    printf("caminos hamiltonianos (secuencias de clases): %llu\n", simple_cnt);
    printf("ponderados por multiplicidad de T: %llu\n", (u64)weighted);
    return 0;
}
