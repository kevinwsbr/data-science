from scipy.stats import normaltest
import matplotlib.pyplot as plt
from scipy.stats import beta
import statistics


def show_hist(x):
    plt.hist(x, 50, density=True, facecolor="b", alpha=0.5)
    plt . title("Histograma")
    plt.show()


def normal_test(x, var):
    sig = 0.05

    stat_test, p_value = normaltest(x)

    res = p_value <= sig
    print(f"{var}: É uma distribuição normal (H0)?", "Não" if res else "Sim")


v1 = beta.rvs(2, 8, size=1000)
v2 = beta.rvs(2, .8, size=1000)

print(
    f"Média de v1: {statistics.mean(v1)}\nMediana de v1: {statistics.median(v1)}")
print(
    f"Média de v1: {statistics.mean(v2)}\nMediana de v2: {statistics.median(v2)}")

show_hist(v1)
normal_test(v1, "v1")

show_hist(v2)
normal_test(v2, "v2")
