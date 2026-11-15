// S está en orden lexicográfico por filas: el índice es el orden en que el backtracking encuentra cada solución.
static int64_t *off; static int32_t *adj;
static void csr(void) {
    size_t n = CNT; off = calloc(n + 2, 8);
    for (size_t e = 0; e < nE; e++) { off[E[e].a + 1]++; off[E[e].b + 1]++; }
    for (size_t i = 0; i < n; i++) off[i + 1] += off[i];
    adj = malloc(2 * nE * 4 + 4); int64_t *p = malloc((n + 1) * 8); memcpy(p, off, (n + 1) * 8);
    for (size_t e = 0; e < nE; e++) { adj[p[E[e].a]++] = E[e].b; adj[p[E[e].b]++] = E[e].a; }
    free(p);
}
static int first_diff(size_t a, size_t b) { for (int i = 0; i < N; i++) if (S[a * 16 + i] != S[b * 16 + i]) return i; return N; }
static uint64_t depth_nodes[17], depth_live[17];
static uint64_t cur_sub;
// subárbol por prefijo de D filas: nodos y soluciones
typedef struct { uint64_t nodes; uint32_t sols; } Px;
static Px *PX; static size_t nPX, capPX; static int CUT;
static uint64_t bt(int r, uint32_t c, uint32_t a, uint32_t b) {       // devuelve soluciones bajo este nodo
    uint32_t F = (1u << N) - 1; depth_nodes[r]++; int rec = r == CUT; uint64_t n0 = 0, sol = 0; size_t id = 0;
    if (rec) { if (nPX == capPX) { capPX = capPX ? capPX * 2 : 1 << 16; PX = realloc(PX, capPX * sizeof(Px)); } id = nPX++; n0 = cur_sub; }
    cur_sub++;
    if (r == N) sol = 1;
    else { uint32_t v = ~(c | a | b) & F; while (v) { uint32_t x = v & -v; v ^= x; sol += bt(r + 1, c | x, ((a | x) << 1) & F, (b | x) >> 1); } }
    if (sol) depth_live[r]++;
    if (rec) { PX[id].nodes = cur_sub - n0; PX[id].sols = (uint32_t)sol; }
    return sol;
}
static void simulate(int D) {
    memset(depth_nodes, 0, sizeof depth_nodes); memset(depth_live, 0, sizeof depth_live); nPX = 0; cur_sub = 0; CUT = D;
    bt(0, 0, 0, 0);
    uint64_t total = cur_sub, base = 0; for (int r = 0; r < D; r++) base += depth_nodes[r];
    uint8_t *found = calloc(CNT, 1); size_t s = 0, nf = 0; uint64_t cost = base, dead = 0, live = 0, ent = 0; int first = 1;
    for (size_t g = 0; g < nPX; g++) {
        if (!PX[g].sols) { cost += 1; dead += PX[g].nodes; continue; }      // sin soluciones: nunca hay predicción
        live++; int hot = first; first = 0;
        for (size_t i = s; i < s + PX[g].sols && !hot; i++) for (int64_t k = off[i]; k < off[i + 1]; k++) if (found[adj[k]]) { hot = 1; break; }
        if (hot) { ent++; cost += PX[g].nodes; for (size_t i = s; i < s + PX[g].sols; i++) { found[i] = 1; nf++; } }
        s += PX[g].sols;
    }
    printf("  D=%2d grupos vivos=%zu entrados=%zu muertos=%zu | nodos %.1f%% del total | soluciones halladas %.2f%%\n", D, (size_t)live, (size_t)ent, nPX - (size_t)live, 100.0 * cost / total, 100.0 * nf / CNT);
    free(found);
}
static void analysis(void) {
    csr();
    uint64_t h[17] = {0}; double sumd = 0; size_t none_earlier = 0, iso = 0;
    for (size_t e = 0; e < nE; e++) { h[first_diff(E[e].a, E[e].b)]++; sumd += (double)(E[e].b - E[e].a) / CNT; }
    for (size_t i = 0; i < CNT; i++) { int any = 0; if (off[i + 1] == off[i]) iso++; for (int64_t k = off[i]; k < off[i + 1]; k++) if ((size_t)adj[k] < i) any = 1; none_earlier += !any; }
    printf("N=%d aristas=%zu distancia media en índice=%.1f%% de Q\n  fila del primer cambio (%%):", N, nE, 100 * sumd / nE);
    for (int r = 0; r < N; r++) printf(" %.1f", 100.0 * h[r] / nE);
    printf("\n  sin vecino anterior: %zu (%.2f%%), de ellas aisladas %zu\n", none_earlier, 100.0 * none_earlier / CNT, iso);
    // grado medio por tramo del orden del backtracking
    printf("  grado medio por décima del orden:"); for (int d = 0; d < 10; d++) { size_t a = CNT * d / 10, b = CNT * (d + 1) / 10; printf(" %.2f", (double)(off[b] - off[a]) / (b - a)); }
    printf("\n");
    for (int D = 2; D < N - 1; D += (N > 12 ? 2 : 1)) simulate(D);
    uint64_t tn = 0, tl = 0; for (int r = 0; r <= N; r++) { tn += depth_nodes[r]; tl += depth_live[r]; }
    printf("  nodos=%llu vivos=%llu (%.1f%%, %.1f vivos/sol)\n  %% muertos por fila:", tn, tl, 100.0 * tl / tn, (double)tl / CNT);
    for (int r = 0; r <= N; r++) printf(" %.0f", depth_nodes[r] ? 100.0 * (depth_nodes[r] - depth_live[r]) / depth_nodes[r] : 0);
    printf("\n");
}
