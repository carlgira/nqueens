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
static void demean_unused(double *v) { double s = 0; for (int32_t i = 0; i < nC; i++) s += v[i]; s /= nC; for (int32_t i = 0; i < nC; i++) v[i] -= s; }
static void construct(uint8_t *q) {   // misma construcción que construct.py, en 0-based
    int e[16], o[16], ne = 0, no = 0; for (int v = 2; v <= N; v += 2) e[ne++] = v; for (int v = 1; v <= N; v += 2) o[no++] = v;
    int r = N % 6;
    if (r == 2) { o[0] = 3; o[1] = 1; int k = 2; for (int i = 3; i < no; i++) o[k++] = o[i]; o[no - 1] = 5; }
    else if (r == 3) { for (int i = 1; i < ne; i++) e[i - 1] = e[i]; e[ne - 1] = 2; for (int i = 2; i < no; i++) o[i - 2] = o[i]; o[no - 2] = 1; o[no - 1] = 3; }
    for (int i = 0; i < ne; i++) q[i] = e[i] - 1; for (int i = 0; i < no; i++) q[ne + i] = o[i] - 1;
}

static int TMAX;
static void analysis(void) {
    size_t n = CNT; int32_t *c = calloc(n, 4); for (size_t i = 0; i < n; i++) c[uf_find(&UFS[0], i)]++;
    int32_t g = 0; for (size_t i = 0; i < n; i++) if (c[i] > c[g]) g = i;
    int32_t *id = malloc(n * 4), *back = malloc(n * 4); nC = 0; for (size_t i = 0; i < n; i++) { id[i] = -1; if (uf_find(&UFS[0], i) == g) { back[nC] = i; id[i] = nC++; } }
    off = calloc(nC + 2, 8); for (size_t e = 0; e < nE; e++) if (id[E[e].a] >= 0) { off[id[E[e].a] + 1]++; off[id[E[e].b] + 1]++; }
    for (int32_t i = 0; i < nC; i++) off[i + 1] += off[i];
    adj = malloc(off[nC] * 4 + 4); int64_t *p = malloc((nC + 1) * 8); memcpy(p, off, (nC + 1) * 8);
    for (size_t e = 0; e < nE; e++) if (id[E[e].a] >= 0) { int a = id[E[e].a], b = id[E[e].b]; adj[p[a]++] = b; adj[p[b]++] = a; }
    dg = malloc(nC * 8); int dmax = 0; double dm = 0, d2 = 0; int32_t top = 0;
    for (int32_t i = 0; i < nC; i++) { dg[i] = off[i + 1] - off[i]; dm += dg[i]; d2 += dg[i] * dg[i]; if (dg[i] > dmax) { dmax = dg[i]; top = i; } }
    dm /= nC; d2 /= nC;
    printf("N=%d C=%d dmax=%d grado medio=%.2f\n", N, nC, dmax, dm);
    printf("  paseo natural (prob proporcional al grado): mas visitada=x%.1f del uniforme, menos=x%.2f, soluciones efectivas=%.1f%%\n", dmax / dm, 1 / dm, 100 * dm * dm / d2);
    printf("  solucion de mayor grado:"); for (int i = 0; i < N; i++) printf(" %d", S[(size_t)back[top] * 16 + i] + 1); printf("\n");
    double h[8] = {0}; for (int32_t i = 0; i < nC; i++) { double r = dg[i] / dm; for (int b = 0; b < 8; b++) if (r < (b + 1) * .5 || b == 7) { h[b] += dg[i]; break; } }
    // estado de minima energia de -A (salto puro): iteracion de potencia sobre A + cI
    double *x = malloc(nC * 8), *y = malloc(nC * 8), *w = malloc(nC * 8); for (int32_t i = 0; i < nC; i++) x[i] = sqrt(dg[i]);
    double lam = 0, nn = sqrt(dot(x, x)); for (int32_t i = 0; i < nC; i++) x[i] /= nn;
    int it; for (it = 0; it < 4000; it++) { Lmul(x, w);
        #pragma omp parallel for
        for (int32_t i = 0; i < nC; i++) y[i] = (dg[i] - 0) * x[i] - w[i] + 1.0 * x[i];
        double l2 = sqrt(dot(y, y)); for (int32_t i = 0; i < nC; i++) x[i] = y[i] / l2; if (fabs(l2 - lam) < 1e-9 * l2) { lam = l2; break; } lam = l2; }
    double ipr = 0, mx = 0, tail = 0; for (int32_t i = 0; i < nC; i++) { double q = x[i] * x[i]; ipr += q * q; if (q > mx) mx = q; }
    double *srt = malloc(nC * 8); for (int32_t i = 0; i < nC; i++) srt[i] = x[i] * x[i];
    int cmpd(const void *, const void *); qsort(srt, nC, 8, cmpd); for (int32_t i = 0; i < nC / 1000 + 1; i++) tail += srt[i];
    printf("  estado de -A: lambda=%.3f iter=%d soluciones efectivas=%.0f (%.3f%% de C) max prob=x%.0f del uniforme, top 0,1%% acumula %.1f%%\n", lam - 1, it, 1 / ipr, 100 / ipr / nC, mx * nC, 100 * tail);
    // onda unitaria desde la solucion construida: Chebyshev sobre H = (L - R)/R, espectro en [-1,1]
    uint8_t q[16] = {0}; construct(q); long si = find_index(q); int32_t s = si >= 0 ? id[si] : -1; if (s < 0) { printf("  construida fuera; uso nodo 0\n"); s = 0; } if (getenv("SALIDA")) { s = atoi(getenv("SALIDA")) % nC; } printf("  salida: grado %.0f\n", dg[s]);
    double R = dmax + 1; int nt = 0; double ts[16]; for (double t = 0.25; t <= TMAX; t *= 4) ts[nt++] = t; int K = (int)(R * TMAX) + 60;
    double *re[16], *im[16]; for (int j = 0; j < nt; j++) { re[j] = calloc(nC, 8); im[j] = calloc(nC, 8); }
    double *a = calloc(nC, 8), *b = malloc(nC * 8), *d = malloc(nC * 8); a[s] = 1; Lmul(a, w); for (int32_t i = 0; i < nC; i++) b[i] = (w[i] - R * a[i]) / R;
    for (int k = 0; k <= K; k++) { double *cur = k == 0 ? a : b;
        for (int j = 0; j < nt; j++) { double co = (k ? 2 : 1) * jn(k, R * ts[j]); int m = k & 3; double *T = (m & 1) ? im[j] : re[j]; double sg = (m == 0 || m == 3) ? 1 : -1; if (m == 1) sg = -1; if (m == 3) sg = 1; if (fabs(co) < 1e-16) continue;
            #pragma omp parallel for
            for (int32_t i = 0; i < nC; i++) T[i] += sg * co * cur[i]; }
        if (k == 0) continue;
        Lmul(b, w);
        #pragma omp parallel for
        for (int32_t i = 0; i < nC; i++) d[i] = 2 * (w[i] - R * b[i]) / R - a[i];
        double *z = a; a = b; b = d; d = z; }
    double *m = calloc(N * N, 8); for (int32_t i = 0; i < nC; i++) for (int r = 0; r < N; r++) m[r * N + S[(size_t)back[i] * 16 + r]] += 1.0 / nC;
    printf("  t  norma  |<u|psi>|^2  sol.efectivas/C  prob inicio  max|M_t-M|/maxM  TV\n");
    for (int j = 0; j < nt; j++) { double nr = 0, ip = 0, ur = 0, ui = 0, tv = 0; double *mt = calloc(N * N, 8);
        for (int32_t i = 0; i < nC; i++) { double pr = re[j][i] * re[j][i] + im[j][i] * im[j][i]; nr += pr; ip += pr * pr; ur += re[j][i]; ui += im[j][i]; tv += fabs(pr - 1.0 / nC); for (int r = 0; r < N; r++) mt[r * N + S[(size_t)back[i] * 16 + r]] += pr; }
        double dM = 0, mM = 0; for (int i = 0; i < N * N; i++) { if (fabs(mt[i] - m[i]) > dM) dM = fabs(mt[i] - m[i]); if (m[i] > mM) mM = m[i]; }
        printf("  %7.2f %.6f %.3e  %.4f  %.3e  %.4f  %.4f\n", ts[j], nr, (ur * ur + ui * ui) / nC, 1 / ip / nC, re[j][s] * re[j][s] + im[j][s] * im[j][s], dM / mM, tv / 2); free(mt); }
}
int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? 1 : x > y ? -1 : 0; }
