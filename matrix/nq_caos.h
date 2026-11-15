// KIT = iteraciones del mapa por paso: cada iteración da 1 bit y elegir entre d vecinos pide log2(d).
static int KIT = 1;
// El mapa de visitados solo se usa para medir: ningún paseo lo consulta.
static int64_t *off; static int32_t *adj; static int32_t *comp; static int32_t nG;
static void build(void) {
    size_t n = CNT; off = calloc(n + 2, 8); for (size_t e = 0; e < nE; e++) { off[E[e].a + 1]++; off[E[e].b + 1]++; }
    for (size_t i = 0; i < n; i++) off[i + 1] += off[i];
    adj = malloc(off[n] * 4 + 4); int64_t *p = malloc((n + 1) * 8); memcpy(p, off, (n + 1) * 8);
    for (size_t e = 0; e < nE; e++) { adj[p[E[e].a]++] = E[e].b; adj[p[E[e].b]++] = E[e].a; } free(p);
}
static uint64_t mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static uint64_t solhash(int32_t a) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < N; i++) h = (h ^ S[(size_t)a * 16 + i]) * 1099511628211ULL; return mix(h); }
static void construct(uint8_t *q) {
    int e[16], o[16], ne = 0, no = 0; for (int v = 2; v <= N; v += 2) e[ne++] = v; for (int v = 1; v <= N; v += 2) o[no++] = v;
    int r = N % 6;
    if (r == 2) { o[0] = 3; o[1] = 1; int k = 2; for (int i = 3; i < no; i++) o[k++] = o[i]; o[no - 1] = 5; }
    else if (r == 3) { for (int i = 1; i < ne; i++) e[i - 1] = e[i]; e[ne - 1] = 2; for (int i = 2; i < no; i++) o[i - 2] = o[i]; o[no - 2] = 1; o[no - 1] = 3; }
    for (int i = 0; i < ne; i++) q[i] = e[i] - 1; for (int i = 0; i < no; i++) q[ne + i] = o[i] - 1;
}
enum { XOR, LOGI, LOGU, TENT, ROT, HASH, ROTOR, NW };
static const char *WN[NW] = {"pseudoaleatorio", "logistico directo", "logistico uniformizado", "rotacion (Sturmian)", "hash(s,t)", "hash(s) sin estado", "rotor (1 contador/nodo)"};
static void walk(int kind, int32_t s, const int32_t *lab, int32_t g) {
    size_t n = CNT; uint8_t *v = calloc(n, 1); uint8_t *rot = kind == ROTOR ? calloc(n, 1) : NULL; size_t back = 0; size_t seen = 1, max = (size_t)40 * nG, marks[4] = {nG / 2, (size_t)(nG * .9), (size_t)(nG * .99), (size_t)(nG * .999)}; int m = 0; size_t at[4] = {0};
    uint64_t x = 0x9E3779B97F4A7C15ULL; double y = 0.1234567891, z = 0.3141592653; const double phi = 0.6180339887498949; v[s] = 1; int32_t a = s, prev = -1; size_t t;
    for (t = 1; t <= max; t++) { int d = off[a + 1] - off[a], i;
        switch (kind) {
            case XOR: x ^= x << 13; x ^= x >> 7; x ^= x << 17; i = (x >> 20) % d; break;
            case LOGI: y = 3.9999 * y * (1 - y); i = (int)(y * d); break;
            case LOGU: for (int q_ = 0; q_ < KIT; q_++) y = 3.9999 * y * (1 - y); i = (int)(asin(sqrt(y)) * 2 / M_PI * d); if (i >= d) i = d - 1; break;
            case TENT: z += phi; z -= (long)z; i = (int)(z * d); break;
            case ROT: i = -1; { uint64_t h = solhash(a) + t; i = mix(h) % d; } break;
            case HASH: i = solhash(a) % d; break;
            default: i = rot[a] % d; rot[a]++; }
        { int32_t pa = a; a = adj[off[a] + i]; if (t >= 2 && a == prev) back++; prev = pa; }
        if (!v[a]) { v[a] = 1; seen++; while (m < 4 && seen >= marks[m]) at[m++] = t; if (m == 4) break; }
        else if (kind == HASH) { break; } }
    printf("   %-24s vuelve atras=%.0f%% visita=%.2f%% | pasos/C hasta 50%%, 90%%, 99%%, 99,9%%:", WN[kind], 100.0 * back / (t > max ? max : t), 100.0 * seen / nG);
    for (int j = 0; j < 4; j++) { if (at[j]) printf(" %.2f", (double)at[j] / nG); else printf(" -"); } printf("\n"); free(v); free(rot);
}
static double *PHI; static const uint8_t *REF;
static void potential(const char *nm, int32_t *lab, int32_t g) {   // padre = vecino de menor phi; raíz = mínimo local
    size_t n = CNT; int32_t *par = malloc(n * 4); size_t roots = 0, iso = 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { int32_t b = i; double m = PHI[i]; for (int64_t k = off[i]; k < off[i + 1]; k++) { int32_t c = adj[k]; if (PHI[c] < m || (PHI[c] == m && c < b)) { m = PHI[c]; b = c; } } par[i] = b; }
    for (int r = 0; r < 64; r++) { int ch = 0;
        #pragma omp parallel for reduction(|:ch)
        for (size_t i = 0; i < n; i++) if (par[par[i]] != par[i]) { par[i] = par[par[i]]; ch = 1; } if (!ch) break; }
    int32_t *sz = calloc(n, 4); for (size_t i = 0; i < n; i++) if (lab[i] == g) sz[par[i]]++;
    int mx = 0; for (size_t i = 0; i < n; i++) if (sz[i]) { roots++; if (sz[i] > mx) mx = sz[i]; }
    printf("   %-26s mínimos locales=%zu (%.2f%% de C), árbol mayor=%d (%.2f%%), media=%.1f\n", nm, roots, 100.0 * roots / nG, mx, 100.0 * mx / nG, (double)nG / roots); free(par); free(sz);
}
static void analysis(void) {
    build(); size_t n = CNT; int32_t *lab = malloc(n * 4), g = 0; int32_t *c = calloc(n, 4);
    for (size_t i = 0; i < n; i++) { lab[i] = uf_find(&UFS[0], i); c[lab[i]]++; } for (size_t i = 0; i < n; i++) if (c[i] > c[g]) g = i; nG = c[g];
    uint8_t q[16] = {0}; construct(q); long s = find_index(q); printf("N=%d Q=%zu C=%d\n", N, n, nG);
    if (s < 0 || lab[s] != g) { for (size_t i = 0; i < n; i++) if (lab[i] == g) { s = i; break; } }
    printf("  paseos sin memoria:\n"); for (int k = 0; k < NW; k++) walk(k, s, lab, g);
    printf("  logistico uniformizado por iteraciones del mapa en cada paso:\n"); for (KIT = 2; KIT <= 8; KIT = KIT < 4 ? KIT + 1 : KIT * 2) { printf("   k=%d", KIT); walk(LOGU, s, lab, g); }
    PHI = malloc(n * 8); REF = S + (size_t)s * 16; printf("  funciones sin memoria:\n");
    for (size_t i = 0; i < n; i++) PHI[i] = i; potential("indice lexicografico", lab, g);
    for (size_t i = 0; i < n; i++) PHI[i] = (double)solhash(i); potential("hash caotico", lab, g);
    for (size_t i = 0; i < n; i++) PHI[i] = -(double)(off[i + 1] - off[i]); potential("mayor grado", lab, g);
    for (size_t i = 0; i < n; i++) { int h = 0; for (int r = 0; r < N; r++) h += S[i * 16 + r] != REF[r]; PHI[i] = h; } potential("Hamming a la referencia", lab, g);
    for (size_t i = 0; i < n; i++) { int t[16], seen[16] = {0}, cy = 0; for (int r = 0; r < N; r++) t[REF[r]] = S[i * 16 + r]; for (int k = 0; k < N; k++) if (!seen[k]) { cy++; int m = k; while (!seen[m]) { seen[m] = 1; m = t[m]; } } PHI[i] = N - cy; } potential("Cayley a la referencia", lab, g);
    for (size_t i = 0; i < n; i++) { double e = 0; for (int r = 0; r < N; r++) e += abs((int)S[i * 16 + r] - (int)REF[r]); PHI[i] = e; } potential("distancia L1 a referencia", lab, g);
    for (size_t i = 0; i < n; i++) { double e = 0; for (int r = 0; r < N; r++) { double d = S[i * 16 + r] - REF[r]; e += d * d; } PHI[i] = e; } potential("distancia L2 a referencia", lab, g);
    // oráculo: distancia BFS real (necesita memoria); referencia de un solo mínimo
    int32_t *d = malloc(n * 4), *Q = malloc(n * 4); memset(d, -1, n * 4); size_t h = 0, tt = 0; Q[tt++] = s; d[s] = 0; while (h < tt) { int32_t x = Q[h++]; for (int64_t k = off[x]; k < off[x + 1]; k++) if (d[adj[k]] < 0) { d[adj[k]] = d[x] + 1; Q[tt++] = adj[k]; } }
    for (size_t i = 0; i < n; i++) PHI[i] = d[i] < 0 ? 1e9 : d[i]; potential("distancia BFS (con memoria)", lab, g);
}
