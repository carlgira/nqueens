// Regla = palabra periódica (t1..tp) de permutaciones T sobre columnas, aplicada una y otra vez desde una solución.
static int PMAX;
static int64_t *off; static int32_t *adj, *tid; static uint64_t *Tk; static int nT; static int32_t *deg_;
static uint64_t tkey(const uint8_t *a, const uint8_t *b) { uint8_t t[16]; for (int c = 0; c < 16; c++) t[c] = c; for (int r = 0; r < N; r++) t[a[r]] = b[r]; uint64_t k = 0; for (int c = 0; c < N; c++) k |= (uint64_t)t[c] << (4 * c); return k; }
static int cmp64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int tindex(uint64_t k) { int lo = 0, hi = nT; while (lo < hi) { int m = (lo + hi) / 2; if (Tk[m] < k) lo = m + 1; else hi = m; } return lo; }
static void csr(void) {
    size_t n = CNT; off = calloc(n + 2, 8); for (size_t e = 0; e < nE; e++) { off[E[e].a + 1]++; off[E[e].b + 1]++; }
    for (size_t i = 0; i < n; i++) off[i + 1] += off[i];
    size_t m = off[n]; adj = malloc(m * 4 + 4); tid = malloc(m * 4 + 4); int64_t *p = malloc((n + 1) * 8); memcpy(p, off, (n + 1) * 8);
    uint64_t *ks = malloc(m * 8 + 8), *es = malloc(m * 8 + 8);
    for (size_t e = 0; e < nE; e++) { const uint8_t *a = S + (size_t)E[e].a * 16, *b = S + (size_t)E[e].b * 16; es[2 * e] = ks[2 * e] = tkey(a, b); es[2 * e + 1] = ks[2 * e + 1] = tkey(b, a); }
    qsort(ks, m, 8, cmp64); Tk = malloc(m * 8 + 8); nT = 0; for (size_t i = 0; i < m; i++) if (!i || ks[i] != ks[i - 1]) Tk[nT++] = ks[i];
    for (size_t e = 0; e < nE; e++) { int x = E[e].a, y = E[e].b; tid[p[x]] = tindex(es[2 * e]); adj[p[x]++] = y; tid[p[y]] = tindex(es[2 * e + 1]); adj[p[y]++] = x; }
    free(ks); free(es); free(p); deg_ = malloc(n * 4); for (size_t i = 0; i < n; i++) deg_[i] = off[i + 1] - off[i];
}
static inline int32_t step(int32_t a, int t) { for (int64_t k = off[a]; k < off[a + 1]; k++) if (tid[k] == t) return adj[k]; return -1; }
#define MAXL 256
static int chain(int32_t s, const int *w, int p, int *closed, int32_t *out) {   // nodos distintos de la cadena
    int32_t a = s; int n = 1, ph = 0; out[0] = s; *closed = 0;
    while (n < MAXL) { int32_t b = step(a, w[ph]); if (b < 0) return n; ph = (ph + 1) % p; if (b == s && ph == 0) { *closed = 1; return n; } for (int i = 0; i < n; i++) if (out[i] == b) return n; out[n++] = b; a = b; }
    return n;
}
static void show(int32_t a) { for (int i = 0; i < N; i++) printf(" %d", S[(size_t)a * 16 + i] + 1); }
static void showT(int t) { uint64_t k = Tk[t]; int seen[16] = {0}; for (int c = 0; c < N; c++) { int d = (k >> (4 * c)) & 15; if (d != c && !seen[c]) { printf("("); int x = c; while (!seen[x]) { seen[x] = 1; printf("%d%s", x, ((k >> (4 * x)) & 15) != c ? " " : ""); x = (k >> (4 * x)) & 15; } printf(")"); } } }
static void chains(int p) {
    long long hist[MAXL + 1] = {0}; int bestL = 0, bestC = 0, bw[4], bc[4]; int32_t bs = -1, bcs = -1; long long closed = 0, walks = 0;
    #pragma omp parallel
    { int32_t out[MAXL]; int w[4], L = 0, Lc = 0, lw[4], lc[4]; int32_t ls = -1, lcs = -1; long long h[MAXL + 1] = {0}, cl = 0, wk = 0;
      #pragma omp for schedule(dynamic, 256) nowait
      for (size_t i = 0; i < CNT; i++) { int32_t a0 = i;
        // primeras p aristas distintas de la inmediata vuelta atrás
        int64_t k1, k2, k3, k4;
        for (k1 = off[a0]; k1 < off[a0 + 1]; k1++) { w[0] = tid[k1]; int32_t a1 = adj[k1];
          int lim2 = p >= 2 ? (int)(off[a1 + 1] - off[a1]) : 1;
          for (int u2 = 0; u2 < lim2; u2++) { int32_t a2 = a1; if (p >= 2) { w[1] = tid[off[a1] + u2]; a2 = adj[off[a1] + u2]; if (a2 == a0) continue; }
            int lim3 = p >= 3 ? (int)(off[a2 + 1] - off[a2]) : 1;
            for (int u3 = 0; u3 < lim3; u3++) { if (p >= 3) { w[2] = tid[off[a2] + u3]; if (adj[off[a2] + u3] == a1) continue; }
              int c, n = chain(a0, w, p, &c, out); h[n]++; wk++; cl += c;
              if (n > L) { L = n; ls = a0; memcpy(lw, w, sizeof w); }
              if (c && n > Lc) { Lc = n; lcs = a0; memcpy(lc, w, sizeof w); } } } } }
      #pragma omp critical
      { for (int i = 0; i <= MAXL; i++) hist[i] += h[i]; closed += cl; walks += wk; if (L > bestL) { bestL = L; bs = ls; memcpy(bw, lw, sizeof lw); } if (Lc > bestC) { bestC = Lc; bcs = lcs; memcpy(bc, lc, sizeof lc); } } }
    printf(" p=%d palabras-inicio=%lld cerradas=%lld | cadena mayor=%d, ciclo mayor=%d\n   hist:", p, walks, closed, bestL, bestC);
    for (int i = 2; i <= MAXL; i++) if (hist[i]) printf(" %d:%lld", i, hist[i]); printf("\n");
    if (bs >= 0) { int c; int32_t o[MAXL]; int n = chain(bs, bw, p, &c, o); double d = 0; for (int i = 0; i < n; i++) d += deg_[o[i]]; printf("   mejor cadena: inicio"); show(bs); printf(" regla"); for (int i = 0; i < p; i++) { printf(" "); showT(bw[i]); } printf(" grado medio=%.2f\n", d / n); }
    if (bcs >= 0) { int c; int32_t o[MAXL]; int n = chain(bcs, bc, p, &c, o); double d = 0; for (int i = 0; i < n; i++) d += deg_[o[i]]; printf("   mejor ciclo: inicio"); show(bcs); printf(" regla"); for (int i = 0; i < p; i++) { printf(" "); showT(bc[i]); } printf(" grado medio=%.2f\n", d / n); }
}
static void cover(void) {     // voraz sobre reglas de periodo 2 (solo N<=14: es caro)
    if (N > 14) return;
    typedef struct { uint64_t k; int32_t c; } WC;
    size_t cap = (size_t)nT * nT; uint32_t *cnt = NULL; if (nT < 12000) cnt = calloc(cap, 4); if (!cnt) { printf(" cobertura omitida\n"); return; }
    for (size_t a = 0; a < CNT; a++) for (int64_t k = off[a]; k < off[a + 1]; k++) { int32_t b = adj[k]; for (int64_t l = off[b]; l < off[b + 1]; l++) if ((size_t)adj[l] != a) cnt[(size_t)tid[k] * nT + tid[l]]++; }
    enum { TOP = 400 }; uint64_t tk[TOP]; uint32_t tc[TOP]; memset(tc, 0, sizeof tc);
    for (size_t i = 0; i < cap; i++) { uint32_t c = cnt[i]; if (c > tc[TOP - 1]) { int j = TOP - 1; while (j > 0 && tc[j - 1] < c) { tc[j] = tc[j - 1]; tk[j] = tk[j - 1]; j--; } tc[j] = c; tk[j] = i; } }
    size_t dist = 0; for (size_t i = 0; i < cap; i++) dist += cnt[i] > 0;
    free(cnt); uint8_t *cov = calloc(CNT, 1); size_t tot = 0; int32_t out[MAXL]; int used[TOP] = {0};
    size_t base = 0; for (size_t i = 0; i < CNT; i++) base += deg_[i] > 0;
    printf(" reglas de periodo 2 distintas=%zu; mayor regla actua en %u soluciones (%.3f%%)\n cobertura voraz:", dist, tc[0], 100.0 * tc[0] / CNT);
    for (int r = 1; r <= 100; r++) { int best = -1; size_t bg = 0;
        for (int t = 0; t < TOP; t++) if (!used[t]) { int w[2] = {(int)(tk[t] / nT), (int)(tk[t] % nT)}; size_t g = 0;
            #pragma omp parallel for reduction(+:g) schedule(static)
            for (size_t a = 0; a < CNT; a++) { int32_t b = step(a, w[0]); if (b < 0) continue; int32_t c = step(b, w[1]); if (c < 0) continue; g += !cov[a]; }
            if (g > bg) { bg = g; best = t; } }
        if (best < 0) break; used[best] = 1; int w[2] = {(int)(tk[best] / nT), (int)(tk[best] % nT)};
        for (size_t a = 0; a < CNT; a++) { int32_t b = step(a, w[0]); if (b < 0) continue; int32_t c = step(b, w[1]); if (c < 0) continue; if (!cov[a]) { cov[a] = 1; tot++; } }
        if (r == 1 || r == 10 || r == 100) printf(" %d reglas=%.2f%%", r, 100.0 * tot / CNT); }
    printf("  (de las 400 mas frecuentes)\n"); free(cov);
}
static void pathcover(void) {  // cadenas sin regla: Warnsdorff desde los nodos de menor grado
    size_t n = CNT; uint8_t *v = calloc(n, 1); int32_t *order = malloc(n * 4), *fr = malloc(n * 4); size_t cnt[64] = {0};
    for (size_t i = 0; i < n; i++) { fr[i] = deg_[i]; cnt[deg_[i] < 63 ? deg_[i] : 63]++; }
    size_t st[64], acc = 0; for (int d = 0; d < 64; d++) { st[d] = acc; acc += cnt[d]; } for (size_t i = 0; i < n; i++) order[st[deg_[i] < 63 ? deg_[i] : 63]++] = i;
    size_t paths = 0, sing = 0, leaves = 0, longest = 0; double sum = 0; for (size_t i = 0; i < n; i++) leaves += deg_[i] == 1;
    for (size_t z = 0; z < n; z++) { int32_t a = order[z]; if (v[a] || !deg_[a]) continue; size_t len = 1; v[a] = 1;
        for (int64_t k = off[a]; k < off[a + 1]; k++) fr[adj[k]]--;
        for (;;) { int32_t b = -1; int bf = 1 << 30; for (int64_t k = off[a]; k < off[a + 1]; k++) { int32_t c = adj[k]; if (!v[c] && fr[c] < bf) { bf = fr[c]; b = c; } } if (b < 0) break; v[b] = 1; for (int64_t k = off[b]; k < off[b + 1]; k++) fr[adj[k]]--; a = b; len++; }
        paths++; sum += len; sing += len == 1; if (len > longest) longest = len; }
    size_t c1 = 0; for (size_t i = 0; i < n; i++) c1 += deg_[i] > 0;
    printf(" cubrir sin regla: caminos=%zu para %zu soluciones, media=%.1f, mayor=%zu, de 1 nodo=%zu; cota inferior por hojas=%zu\n", paths, c1, sum / paths, longest, sing, leaves / 2);
    free(v); free(order); free(fr);
}
typedef struct { uint64_t k; int32_t i; } KI;
static int cmpki(const void *a, const void *b) { const KI *x = a, *y = b; return x->k < y->k ? -1 : x->k > y->k ? 1 : (x->i < y->i ? -1 : x->i > y->i); }
static int32_t *up; static int32_t fu(int32_t x) { while (up[x] != x) { up[x] = up[up[x]]; x = up[x]; } return x; }
static void superstring(void) {  // voraz clásico: solapar el final de una con el principio de otra
    size_t n = CNT; uint8_t *ht = calloc(n, 1), *tt = calloc(n, 1); up = malloc(n * 4); for (size_t i = 0; i < n; i++) up[i] = i; KI *h = malloc(n * sizeof(KI)); size_t tot = 0, byk[17] = {0};
    for (int k = N - 1; k >= 1; k--) { size_t m = 0; for (size_t i = 0; i < n; i++) if (!ht[i]) { uint64_t key = 0; for (int j = 0; j < k; j++) key = key * 16 + S[i * 16 + j]; h[m].k = key; h[m].i = i; m++; }
        qsort(h, m, sizeof(KI), cmpki); int8_t *dead = calloc(m, 1);
        for (size_t a = 0; a < n; a++) if (!tt[a]) { uint64_t key = 0; for (int j = N - k; j < N; j++) key = key * 16 + S[a * 16 + j]; size_t lo = 0, hi = m; while (lo < hi) { size_t md = (lo + hi) / 2; if (h[md].k < key) lo = md + 1; else hi = md; }
            for (size_t x = lo; x < m && h[x].k == key; x++) if (!ht[h[x].i] && fu(a) != fu(h[x].i)) { tt[a] = 1; ht[h[x].i] = 1; up[fu(a)] = fu(h[x].i); tot += k; byk[k]++; break; } }
        free(dead); }
    size_t L = N * n - tot; printf(" superstring voraz: longitud=%zu de %zu (%.1f%%), %.2f dígitos nuevos por solución; empalmes por solape:", L, N * n, 100.0 * L / (N * n), (double)L / n); for (int k = N - 1; k >= 1; k--) if (byk[k]) printf(" %d:%zu", k, byk[k]); printf("\n");
}
static void attractors(void) {
    size_t n = CNT; size_t *c = calloc(n, sizeof(size_t)); for (size_t i = 0; i < n; i++) c[uf_find(&UFS[0], i)]++; size_t g = 0; for (size_t i = 0; i < n; i++) if (c[i] > c[g]) g = i;
    double *x = malloc(n * 8), *y = malloc(n * 8); for (size_t i = 0; i < n; i++) x[i] = uf_find(&UFS[0], i) == g ? sqrt(deg_[i]) + 1 : 0;
    for (int it = 0; it < 3000; it++) { double s = 0;
        #pragma omp parallel for reduction(+:s)
        for (size_t i = 0; i < n; i++) { double v = x[i]; for (int64_t k = off[i]; k < off[i + 1]; k++) v += x[adj[k]]; y[i] = v; s += v * v; }
        s = sqrt(s); for (size_t i = 0; i < n; i++) x[i] = y[i] / s; }
    int32_t top[6] = {-1, -1, -1, -1, -1, -1}; for (size_t i = 0; i < n; i++) { double v = x[i]; int32_t id = i; for (int t = 0; t < 6; t++) if (top[t] < 0 || v > x[top[t]]) { int32_t o = top[t]; top[t] = id; id = o; if (id < 0) break; v = x[id]; } }
    size_t lin = 0, lintop = 0; double pl = 0; for (size_t i = 0; i < n; i++) { int ok = 0; for (int a = 1; a <= N && !ok; a++) { ok = 1; for (int r = 0; r < N; r++) if ((S[i * 16 + r] + 1) != (a * (r + 1)) % (N + 1)) { ok = 0; break; } } if (ok) { lin++; pl += x[i] * x[i]; } }
    printf(" atractor -A, 6 picos:\n"); for (int t = 0; t < 6; t++) { printf("   p=%.4f grado=%d:", x[top[t]] * x[top[t]], deg_[top[t]]); show(top[t]); printf("\n"); }
    printf("   lineales mod N+1: %zu, prob conjunta=%.4f\n", lin, pl); free(x); free(y); free(c);
}
static void analysis(void) { csr(); printf("N=%d Q=%zu T distintas=%d\n", N, CNT, nT); attractors(); pathcover(); superstring(); for (int p = 1; p <= PMAX; p++) chains(p); cover(); }
