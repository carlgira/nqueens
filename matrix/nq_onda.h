// Sobre el componente mayor C: L = D - A. El vector uniforme u de C cumple L u = 0 (la onda buscada).
// P = I - L/(dmax+1) es un paseo con distribución estacionaria uniforme.
static int64_t *off; static int32_t *adj; static double *dg; static int32_t nC;
static void Lmul(const double *x, double *y) {          // y = L x
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nC; i++) { double s = dg[i] * x[i]; for (int64_t k = off[i]; k < off[i + 1]; k++) s -= x[adj[k]]; y[i] = s; }
}
static double dot(const double *a, const double *b) { double s = 0; 
    #pragma omp parallel for reduction(+:s)
    for (int32_t i = 0; i < nC; i++) s += a[i] * b[i]; return s; }
static void demean(double *v) { double s = 0; for (int32_t i = 0; i < nC; i++) s += v[i]; s /= nC; for (int32_t i = 0; i < nC; i++) v[i] -= s; }
static uint64_t xs = 0x9E3779B97F4A7C15ULL;
static uint64_t xr(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs; }
static int sturm(const double *a, const double *b, int m, double x) { int c = 0; double d = 1; for (int i = 0; i < m; i++) { d = a[i] - x - (i ? b[i - 1] * b[i - 1] / d : 0); if (d == 0) d = 1e-300; if (d < 0) c++; } return c; }
static double smallest(const double *a, const double *b, int m) { double lo = 0, hi = 1e3; for (int i = 0; i < 200; i++) { double x = (lo + hi) / 2; if (sturm(a, b, m, x) >= 1) hi = x; else lo = x; } return hi; }
static double fiedler(double *out, int m) {
    double *v = malloc(nC * 8), *w = malloc(nC * 8), *p = calloc(nC, 8), *v0 = malloc(nC * 8), *al = malloc(m * 8), *be = malloc(m * 8);
    for (int32_t i = 0; i < nC; i++) v[i] = (double)(xr() >> 11) / (1ULL << 53) - .5; demean(v); double n = sqrt(dot(v, v)); for (int32_t i = 0; i < nC; i++) v[i] /= n; memcpy(v0, v, nC * 8);
    double bp = 0; int M = m;
    for (int j = 0; j < m; j++) { Lmul(v, w); double a = dot(v, w); al[j] = a; for (int32_t i = 0; i < nC; i++) w[i] -= a * v[i] + bp * p[i]; demean(w); double b = sqrt(dot(w, w)); be[j] = b; if (b < 1e-12) { M = j + 1; break; } memcpy(p, v, nC * 8); for (int32_t i = 0; i < nC; i++) v[i] = w[i] / b; bp = b; }
    double lam = smallest(al, be, M);
    double *y = malloc(M * 8), *c = malloc(M * 8), *r = malloc(M * 8); for (int i = 0; i < M; i++) y[i] = 1;     // iteración inversa sobre la tridiagonal
    for (int it = 0; it < 6; it++) { double sh = lam - 1e-10; c[0] = be[0] / (al[0] - sh); r[0] = y[0] / (al[0] - sh);
        for (int i = 1; i < M; i++) { double d = al[i] - sh - be[i - 1] * c[i - 1]; c[i] = be[i] / d; r[i] = (y[i] - be[i - 1] * r[i - 1]) / d; }
        y[M - 1] = r[M - 1]; for (int i = M - 2; i >= 0; i--) y[i] = r[i] - c[i] * y[i + 1]; double s = 0; for (int i = 0; i < M; i++) s += y[i] * y[i]; s = sqrt(s); for (int i = 0; i < M; i++) y[i] /= s; }
    memcpy(v, v0, nC * 8); memset(p, 0, nC * 8); memset(out, 0, nC * 8); bp = 0;
    for (int j = 0; j < M; j++) { for (int32_t i = 0; i < nC; i++) out[i] += y[j] * v[i]; Lmul(v, w); for (int32_t i = 0; i < nC; i++) w[i] -= al[j] * v[i] + bp * p[i]; demean(w); memcpy(p, v, nC * 8); if (be[j] > 1e-12) for (int32_t i = 0; i < nC; i++) v[i] = w[i] / be[j]; bp = be[j]; }
    double s = sqrt(dot(out, out)); for (int32_t i = 0; i < nC; i++) out[i] /= s;
    Lmul(out, w); double res = 0; for (int32_t i = 0; i < nC; i++) res += (w[i] - lam * out[i]) * (w[i] - lam * out[i]);
    printf("  lambda2=%.6f residuo=%.2e iter=%d\n", lam, sqrt(res), M); free(v); free(w); free(p); free(v0); free(al); free(be); free(y); free(c); free(r); return lam;
}
static void construct(uint8_t *q) {   // misma construcción que construct.py, en 0-based
    int e[16], o[16], ne = 0, no = 0; for (int v = 2; v <= N; v += 2) e[ne++] = v; for (int v = 1; v <= N; v += 2) o[no++] = v;
    int r = N % 6;
    if (r == 2) { o[0] = 3; o[1] = 1; int k = 2; for (int i = 3; i < no; i++) o[k++] = o[i]; o[no - 1] = 5; }
    else if (r == 3) { for (int i = 1; i < ne; i++) e[i - 1] = e[i]; e[ne - 1] = 2; for (int i = 2; i < no; i++) o[i - 2] = o[i]; o[no - 2] = 1; o[no - 1] = 3; }
    for (int i = 0; i < ne; i++) q[i] = e[i] - 1; for (int i = 0; i < no; i++) q[ne + i] = o[i] - 1;
}
static void analysis(void) {
    size_t n = CNT; int32_t *c = calloc(n, 4); for (size_t i = 0; i < n; i++) c[uf_find(&UFS[0], i)]++;
    int32_t g = 0; for (size_t i = 0; i < n; i++) if (c[i] > c[g]) g = i;
    int32_t *id = malloc(n * 4), *back = malloc(n * 4); nC = 0; for (size_t i = 0; i < n; i++) { id[i] = -1; if (uf_find(&UFS[0], i) == g) { back[nC] = i; id[i] = nC++; } }
    off = calloc(nC + 2, 8); for (size_t e = 0; e < nE; e++) if (id[E[e].a] >= 0) { off[id[E[e].a] + 1]++; off[id[E[e].b] + 1]++; }
    for (int32_t i = 0; i < nC; i++) off[i + 1] += off[i];
    adj = malloc(off[nC] * 4 + 4); int64_t *p = malloc((nC + 1) * 8); memcpy(p, off, (nC + 1) * 8);
    for (size_t e = 0; e < nE; e++) if (id[E[e].a] >= 0) { int a = id[E[e].a], b = id[E[e].b]; adj[p[a]++] = b; adj[p[b]++] = a; }
    dg = malloc(nC * 8); int dmax = 0; for (int32_t i = 0; i < nC; i++) { dg[i] = off[i + 1] - off[i]; if (dg[i] > dmax) dmax = dg[i]; }
    printf("N=%d Q=%zu C=%d (%.2f%%) dmax=%d\n", N, n, nC, 100.0 * nC / n, dmax);
    double *f = malloc(nC * 8); double lam = fiedler(f, 400);
    double ipr = 0, par = 0; for (int32_t i = 0; i < nC; i++) { ipr += f[i] * f[i] * f[i] * f[i]; par += f[i] * (odd_perm(S + (size_t)back[i] * 16) ? -1 : 1); }
    int32_t top = 0; for (int32_t i = 0; i < nC; i++) if (fabs(f[i]) > fabs(f[top])) top = i;
    printf("  fiedler: nodos efectivos(1/IPR)=%.1f  solape con paridad=%.2e  grado del pico=%.0f  gap P=%.5f\n", 1 / ipr, fabs(par) / sqrt(nC), dg[top], lam / (dmax + 1));
    uint8_t q[16] = {0}; construct(q); long si = find_index(q); int32_t s = si >= 0 ? id[si] : -1;
    if (s < 0) { printf("  construida fuera del componente mayor; uso el nodo 0\n"); s = 0; }
    double *m = malloc(N * N * 8), *mt = malloc(N * N * 8); memset(m, 0, N * N * 8);
    for (int32_t i = 0; i < nC; i++) for (int r = 0; r < N; r++) m[r * N + S[(size_t)back[i] * 16 + r]] += 1.0 / nC;
    double *x = calloc(nC, 8), *y = malloc(nC * 8), *w = malloc(nC * 8); x[s] = 1; double a = 1.0 / (dmax + 1), un = 1.0 / nC; int t = 0, next = 1, t25 = -1, t01 = -1;
    printf("  t  TV(uniforme)  max|M_t-M|/max M  fidelidad(t/2)\n");
    while (t < 20000) { Lmul(x, w);
        #pragma omp parallel for
        for (int32_t i = 0; i < nC; i++) y[i] = x[i] - a * w[i];
        double *z = x; x = y; y = z; t++;
        if (t == next || t % 2048 == 0) { next = t * 2; double tv = 0; memset(mt, 0, N * N * 8);
            for (int32_t i = 0; i < nC; i++) { tv += fabs(x[i] - un); for (int r = 0; r < N; r++) mt[r * N + S[(size_t)back[i] * 16 + r]] += x[i]; } tv /= 2;
            double dM = 0, mM = 0; for (int i = 0; i < N * N; i++) { if (fabs(mt[i] - m[i]) > dM) dM = fabs(mt[i] - m[i]); if (m[i] > mM) mM = m[i]; }
            printf("  %5d %.5f  %.5f  %.3e\n", t, tv, dM / mM, 1.0 / (nC * x[s])); fflush(stdout);
            if (t25 < 0 && tv < .25) t25 = t; if (t01 < 0 && tv < .01) { t01 = t; break; } } }
    printf("  mezcla: TV<1/4 en t<=%d, TV<1/100 en t<=%d\n", t25, t01);
}
