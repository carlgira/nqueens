// Hamiltonianos exactos de la componente de 46 (N=8, involución). Entrada: salida de export_componente.py.
#include <stdio.h>
#include <stdlib.h>
typedef unsigned long long u64;
int n, D[64], ad[64][8];
static u64 dfs(int v, u64 seen, int depth) {
    if (depth == n) return 1;
    int ends = 0;
    for (int u = 0; u < n; u++) if (!(seen >> u & 1)) { int f = 0; for (int i = 0; i < D[u]; i++) { int w = ad[u][i]; if (w == v || !(seen >> w & 1)) f++; } if (f == 0) return 0; if (f == 1 && ++ends > 2) return 0; }
    u64 c = 0; for (int i = 0; i < D[v]; i++) { int u = ad[v][i]; if (!(seen >> u & 1)) c += dfs(u, seen | 1ULL << u, depth + 1); }
    return c;
}
int main(int ac, char **av) { FILE *f = fopen(av[1], "r"); int a, b, r; fscanf(f, "%d", &n); while (fscanf(f, "%d %d %d", &a, &b, &r) == 3) ad[a][D[a]++] = b;
    // two levels of tasks for balance
    int S[4096][3], ns = 0; for (int s = 0; s < n; s++) for (int i = 0; i < D[s]; i++) for (int j = 0; j < D[ad[s][i]]; j++) { int m = ad[s][i], t = ad[m][j]; if (t == s) continue; S[ns][0] = s; S[ns][1] = m; S[ns][2] = t; ns++; }
    u64 tot = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:tot)
    for (int i = 0; i < ns; i++) tot += dfs(S[i][2], 1ULL << S[i][0] | 1ULL << S[i][1] | 1ULL << S[i][2], 3);
    printf("H(46, todos los generadores)=%llu\n", tot);
}
