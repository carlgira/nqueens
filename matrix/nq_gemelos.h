// degree stats, configuration-model null, BFS distances
static uint64_t xs = 0x9E3779B97F4A7C15ULL;
static uint64_t xr(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs; }
static int32_t *uu;
static int32_t ff(int32_t x) { while (uu[x] != x) { uu[x] = uu[uu[x]]; x = uu[x]; } return x; }
static void comps(size_t n, size_t *iso, size_t *g1, size_t *g2, int32_t *deg) {
    int32_t *c = calloc(n, 4); *iso = 0; *g1 = 0; *g2 = 0;
    for (size_t i = 0; i < n; i++) c[ff(i)]++;
    for (size_t i = 0; i < n; i++) if (c[i]) { size_t v = c[i]; if (v == 1) (*iso)++; if (v > *g1) { *g2 = *g1; *g1 = v; } else if (v > *g2) *g2 = v; }
    free(c);
}
static void realvsnull(const char *nm, int mask) {           // mask of types
    size_t n = CNT; int32_t *deg = calloc(n, 4); size_t m = 0;
    uu = malloc(n * 4); for (size_t i = 0; i < n; i++) uu[i] = i;
    for (size_t e = 0; e < nE; e++) if (mask >> E[e].t & 1) { deg[E[e].a]++; deg[E[e].b]++; m++; int32_t x = ff(E[e].a), y = ff(E[e].b); if (x != y) uu[x] = y; }
    size_t iso, g1, g2; comps(n, &iso, &g1, &g2, deg);
    printf("  %-14s real: giant=%zu (%.2f%%) second=%zu iso=%zu (%.2f%%)", nm, g1, 100.0 * g1 / n, g2, iso, 100.0 * iso / n);
    int32_t *st = malloc(2 * m * 4); size_t k = 0;
    for (size_t i = 0; i < n; i++) for (int j = 0; j < deg[i]; j++) st[k++] = i;
    for (size_t i = k - 1; i > 0; i--) { size_t j = xr() % (i + 1); int32_t t = st[i]; st[i] = st[j]; st[j] = t; }
    for (size_t i = 0; i < n; i++) uu[i] = i;
    for (size_t e = 0; e + 1 < k; e += 2) { int32_t x = ff(st[e]), y = ff(st[e + 1]); if (x != y) uu[x] = y; }
    comps(n, &iso, &g1, &g2, deg);
    printf("  | null(same degrees): giant=%zu (%.2f%%) second=%zu iso=%zu\n", g1, 100.0 * g1 / n, g2, iso);
    // Poisson expectation of iso
    double dm = 2.0 * m / n, v = 0; for (size_t i = 0; i < n; i++) v += (deg[i] - dm) * (deg[i] - dm); v /= n;
    printf("  %-14s mean degree=%.3f var=%.3f\n", "", dm, v);
    free(st); free(deg); free(uu);
}
static void bfs_stats(int nseeds) {
    size_t n = CNT; int64_t *off = calloc(n + 2, 8);
    for (size_t e = 0; e < nE; e++) { off[E[e].a + 1]++; off[E[e].b + 1]++; }
    for (size_t i = 0; i < n; i++) off[i + 1] += off[i];
    int32_t *adj = malloc(2 * nE * 4); int64_t *p = malloc((n + 1) * 8); memcpy(p, off, (n + 1) * 8);
    for (size_t e = 0; e < nE; e++) { adj[p[E[e].a]++] = E[e].b; adj[p[E[e].b]++] = E[e].a; }
    int32_t *d = malloc(n * 4), *Q = malloc(n * 4);
    int done = 0, tries = 0, maxe = 0; double avg = 0, med = 0; size_t reach = 0;
    while (done < nseeds && tries++ < 1000) {
        int32_t s = xr() % n; memset(d, -1, n * 4); size_t h = 0, t = 0; Q[t++] = s; d[s] = 0;
        while (h < t) { int32_t x = Q[h++]; for (int64_t i = off[x]; i < off[x + 1]; i++) if (d[adj[i]] < 0) { d[adj[i]] = d[x] + 1; Q[t++] = adj[i]; } }
        if (t < n / 2) continue;
        double sum = 0; for (size_t i = 0; i < t; i++) sum += d[Q[i]];
        if (d[Q[t - 1]] > maxe) maxe = d[Q[t - 1]]; avg += sum / t; med += d[Q[t / 2]]; reach = t; done++;
    }
    double k = 2.0 * nE / n, k2 = 0; for (size_t i = 0; i < n; i++) { double g = off[i + 1] - off[i]; k2 += g * g; } k2 /= n;
    printf("  BFS %d seeds in giant (%zu): mean dist=%.2f median=%.1f max ecc=%d | ln(n)/ln(<k2-k>/<k>)=%.2f\n", done, reach, avg / done, med / done, maxe, log((double)n) / log((k2 - k) / k));
    free(adj); free(p); free(off); free(d); free(Q);
}
static void top2(void) {   // two largest comps of 'all': which D4 images stay inside
    size_t n = CNT; int32_t *c = calloc(n, 4); for (size_t i = 0; i < n; i++) c[uf_find(&UFS[0], i)]++;
    int32_t a = -1, b = -1; for (size_t i = 0; i < n; i++) if (c[i]) { if (a < 0 || c[i] > c[a]) { b = a; a = i; } else if (b < 0 || c[i] > c[b]) b = i; }
    printf("  top comps %d %d;", c[a], b >= 0 ? c[b] : 0);
    const char *nm[8] = {"id", "rowflip", "colflip", "rot180", "transp", "t.rflip", "t.cflip", "t.rot"};
    size_t in[8] = {0}, odd = 0, tot = 0;
    for (size_t i = 0; i < n; i++) if (uf_find(&UFS[0], i) == a) { tot++; odd += odd_perm(S + i * 16); uint8_t im[8][16]; images(S + i * 16, im); for (int k = 0; k < 8; k++) { uint8_t t[16] = {0}; memcpy(t, im[k], N); in[k] += uf_find(&UFS[0], find_index(t)) == a; } }
    printf(" biggest: even=%zu odd=%zu; fraction of its nodes whose image stays inside:", tot - odd, odd);
    for (int k = 1; k < 8; k++) printf(" %s=%.3f", nm[k], (double)in[k] / tot);
    printf("\n"); free(c);
}

// two largest components for an edge selection, and what each D4 element does with them
static void twins(const char *nm, int mask) {
    size_t n = CNT; int32_t *c = calloc(n, 4), *lab = malloc(n * 4);
    uu = malloc(n * 4); for (size_t i = 0; i < n; i++) uu[i] = i;
    for (size_t e = 0; e < nE; e++) if (mask >> E[e].t & 1) { int32_t x = ff(E[e].a), y = ff(E[e].b); if (x != y) uu[x] = y; }
    for (size_t i = 0; i < n; i++) { lab[i] = ff(i); c[lab[i]]++; }
    int32_t a = -1, b = -1; for (size_t i = 0; i < n; i++) if (c[i]) { if (a < 0 || c[i] > c[a]) { b = a; a = i; } else if (b < 0 || c[i] > c[b]) b = i; }
    size_t oa = 0, ob = 0; size_t st[8] = {0}, to[8] = {0};
    for (size_t i = 0; i < n; i++) { int isa = lab[i] == a, isb = b >= 0 && lab[i] == b; if (!isa && !isb) continue;
        if (isa) oa += odd_perm(S + i * 16); else ob += odd_perm(S + i * 16);
        if (isa) { uint8_t im[8][16]; images(S + i * 16, im); for (int k = 1; k < 8; k++) { uint8_t t[16] = {0}; memcpy(t, im[k], N); int32_t l = lab[find_index(t)]; st[k] += l == a; to[k] += l == b; } } }
    printf("%2d %-13s top=%d/%d (%.1f%%+%.1f%%) par/impar A=%zu/%zu B=%zu/%zu  ", N, nm, c[a], b >= 0 ? c[b] : 0, 100.0 * c[a] / n, b >= 0 ? 100.0 * c[b] / n : 0, c[a] - oa, oa, b >= 0 ? c[b] - ob : 0, ob);
    const char *nmk[8] = {"", "F", "C", "R180", "T", "Tf", "Tc", "Tr"};
    printf("cambian:"); for (int k = 1; k < 8; k++) if (to[k] * 2 > (size_t)c[a]) printf(" %s", nmk[k]);
    printf(" mezclan:"); for (int k = 1; k < 8; k++) if (to[k] * 2 <= (size_t)c[a] && st[k] != (size_t)c[a]) printf(" %s(%.2f)", nmk[k], (double)st[k] / c[a]);
    printf("\n"); free(c); free(lab); free(uu);
}
static void analysis(void) {
    twins("cayley<=2", 2 | 4 | 8); twins("ciclo3", 4); twins("doble", 8); twins("ciclo3+doble", 4 | 8);
    if (N <= 14) {   // involution T (any support), same exclusions
        size_t base = nE;
        #pragma omp parallel for schedule(dynamic, 8)
        for (size_t i = 0; i < CNT; i++) { const uint8_t *x = S + i * 16; uint8_t inv[16]; for (int k = 0; k < N; k++) inv[x[k]] = k;
            uint8_t im[8][16]; images(x, im); int tor = is_tor(x);
            for (size_t j = i + 1; j < CNT; j++) { const uint8_t *y = S + j * 16; int ok = 1; for (int k = 0; k < N; k++) if (y[inv[y[k]]] != x[k]) { ok = 0; break; }
                if (!ok) continue; int sym = 0; for (int s = 0; s < 8; s++) if (!memcmp(im[s], y, N)) sym = 1; if (sym || (tor && is_tor(y))) continue;
                #pragma omp critical
                { if (nE == capE) { capE *= 2; E = realloc(E, capE * sizeof(Ed)); } E[nE].a = i; E[nE].b = j; E[nE].t = 4; nE++; } } }
        printf("   involucion aristas=%zu grado=%.2f\n", nE - base, 2.0 * (nE - base) / CNT);
        twins("involucion", 16);
    }
}
