"""
Espectro de complejidad de Prony para N=7, 11, 13 (datos medidos con prony_sweep).
Muestra como, al crecer N, la escalera de p (nº de espirales generadoras) se
rellena — salvo p=2, que nunca aparece.
"""
import matplotlib.pyplot as plt

# distribuciones medidas: {p: nº soluciones}. maxp = N//2; p=maxp+1 = irreducible.
DATA = {
    7:  {1: 28, 4: 12},                          # maxp=3 -> irred en p=4
    11: {1: 88, 5: 10, 6: 2582},                 # maxp=5 -> irred en p=6
    13: {1: 130, 3: 18, 4: 4, 5: 30, 6: 70, 7: 73460},  # maxp=6 -> irred en p=7
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, N in zip(axes, (7, 11, 13)):
    d = DATA[N]
    maxp = N // 2
    ps = list(range(1, maxp + 2))
    counts = [d.get(p, 0) for p in ps]
    colors = []
    for p in ps:
        if p == 1:
            colors.append("#1f77b4")        # espiral pura
        elif p == maxp + 1:
            colors.append("#d62728")        # irreducible
        else:
            colors.append("#2ca02c")        # intermedio
    bars = ax.bar(ps, counts, color=colors)
    for p, c in zip(ps, counts):
        if c:
            ax.text(p, c, str(c), ha="center", va="bottom", fontsize=8)
    # marcar el hueco p=2
    ax.axvspan(1.5, 2.5, color="0.9", zorder=0)
    ax.text(2, max(counts) ** 0.5 if max(counts) > 1 else 0.5,
            "p=2\nnunca", ha="center", va="center", fontsize=7, color="0.4")
    ax.set_yscale("log")
    ax.set_xticks(ps)
    ax.set_xlabel("p = nº de espirales generadoras")
    ax.set_title(f"N = {N}   ({sum(d.values())} soluciones)")
    if N == 7:
        ax.set_ylabel("nº de soluciones (log)")

fig.suptitle("Espectro de complejidad de Prony: azul=espiral pura · verde=intermedio · rojo=irreducible\n"
             "al crecer N la escalera se rellena (3,4,5,6...) — pero p=2 nunca aparece",
             fontsize=11)
fig.subplots_adjust(top=0.78, bottom=0.13, wspace=0.25)
fig.savefig("prony_spectrum.png", dpi=130)
print("Guardado: prony_spectrum.png")
