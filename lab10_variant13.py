"""
Лабораторная работа № 10. Вариант 13.
Кугушева Арина, группа БИ2-1.
Дисциплина: Теория алгоритмов.

Алгоритм Дейкстры (линейный поиск минимума и двоичная куча), восстановление
путей, топологическая сортировка и метод критического пути.

Условия варианта 13:
  * разреженные графы, m ≈ 4n, n = 2000; 4000; 8000, веса рёбер 1…100;
  * проект из 12 работ;
  * К6: обнаружение цикла в сетевом графике и сообщение о некорректных
    данных (проверка на графике с искусственно внесённым циклом).

Запуск:
    python lab10_variant13.py                       # все эксперименты
    python lab10_variant13.py --plots plots         # + график в каталог plots
    python lab10_variant13.py --json results.json   # + результаты в JSON
    python lab10_variant13.py --repeats 3           # число повторов замера
"""
import argparse
import gc
import heapq
import json
import random
import statistics
import time
from collections import deque

INF = float("inf")
SEED = 42
SIZES = [2000, 4000, 8000]
DEGREE = 4              # m ≈ 4n
MAX_W = 100


# ====================================================================
# Алгоритм Дейкстры
# ====================================================================
def dijkstra_array(adj, s):
    """Дейкстра с линейным поиском минимума. Трудоёмкость Θ(n²)."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    used = [False] * n
    dist[s] = 0
    for _ in range(n):
        v, best = -1, INF
        for i in range(n):                      # линейный поиск минимума
            if not used[i] and dist[i] < best:
                v, best = i, dist[i]
        if v == -1:                             # остальные вершины недостижимы
            break
        used[v] = True
        for u, w in adj[v]:                     # релаксация рёбер
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
    return dist, parent


def dijkstra_heap(adj, s):
    """Дейкстра с приоритетной очередью (heapq). Трудоёмкость O(m log n)."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:                         # устаревшая запись
            continue
        for u, w in adj[v]:
            if d + w < dist[u]:
                dist[u] = d + w
                parent[u] = v
                heapq.heappush(heap, (dist[u], u))
    return dist, parent


def dijkstra_trace(adj, s):
    """Дейкстра с протоколом шагов (для ручного расчёта на малом графе)."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    used = [False] * n
    dist[s] = 0
    steps = []
    for _ in range(n):
        v, best = -1, INF
        for i in range(n):
            if not used[i] and dist[i] < best:
                v, best = i, dist[i]
        if v == -1:
            break
        used[v] = True
        for u, w in adj[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
        steps.append({"v": v, "d": dist[v], "dist": list(dist), "parent": list(parent)})
    return dist, parent, steps


def restore_path(parent, target):
    """Восстановление пути по массиву предков."""
    path = []
    while target != -1:
        path.append(target)
        target = parent[target]
    return path[::-1]


def generate_graph(n, m, max_weight=MAX_W, seed=SEED):
    """Случайный ориентированный взвешенный граф: n вершин, ~m рёбер.

    Сначала строится остовное дерево (все вершины достижимы из 0),
    затем добавляются случайные рёбра до общего числа m.
    """
    rng = random.Random(seed)
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        adj[rng.randrange(i)].append((i, rng.randint(1, max_weight)))
    for _ in range(max(0, m - (n - 1))):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            adj[u].append((v, rng.randint(1, max_weight)))
    return adj


def count_edges(adj):
    return sum(len(a) for a in adj)


# ====================================================================
# Топологическая сортировка и критический путь
# ====================================================================
def topological_order(tasks):
    """Топологическая сортировка (алгоритм Кана), tasks: код → (t, [пред.]).

    К6: при наличии цикла выбрасывается ValueError со списком работ,
    которые не удалось упорядочить.
    """
    for k, (_, preds) in tasks.items():
        for p in preds:
            if p not in tasks:
                raise ValueError(f"Некорректные данные: работа {k} ссылается "
                                 f"на несуществующего предшественника {p}")
    indeg = {k: len(v[1]) for k, v in tasks.items()}
    succ = {k: [] for k in tasks}
    for k, (_, preds) in tasks.items():
        for p in preds:
            succ[p].append(k)
    queue = deque(sorted(k for k, d in indeg.items() if d == 0))
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(u)
    if len(order) != len(tasks):
        stuck = sorted(k for k in tasks if k not in set(order))
        raise ValueError("Некорректные исходные данные: сетевой график содержит "
                         "цикл. Работы, вошедшие в цикл или зависящие от него: "
                         + ", ".join(stuck))
    return order, succ


def find_cycle(tasks):
    """Находит один конкретный цикл (список работ) или None. Поиск в глубину."""
    succ = {k: [] for k in tasks}
    for k, (_, preds) in tasks.items():
        for p in preds:
            if p in succ:
                succ[p].append(k)
    color = {k: 0 for k in tasks}               # 0 — белая, 1 — серая, 2 — чёрная
    stack_path = []

    def dfs(v):
        color[v] = 1
        stack_path.append(v)
        for u in succ[v]:
            if color[u] == 1:
                return stack_path[stack_path.index(u):] + [u]
            if color[u] == 0:
                res = dfs(u)
                if res:
                    return res
        stack_path.pop()
        color[v] = 2
        return None

    for k in sorted(tasks):
        if color[k] == 0:
            res = dfs(k)
            if res:
                return res
    return None


def critical_path(tasks):
    """Расчёт ES, EF, LS, LF, резервов и критических работ."""
    order, succ = topological_order(tasks)
    ES, EF = {}, {}
    for v in order:                             # прямой проход
        duration, preds = tasks[v]
        ES[v] = max((EF[p] for p in preds), default=0)
        EF[v] = ES[v] + duration
    T = max(EF.values())

    LS, LF = {}, {}
    for v in reversed(order):                   # обратный проход
        duration = tasks[v][0]
        LF[v] = min((LS[u] for u in succ[v]), default=T)
        LS[v] = LF[v] - duration

    reserve = {v: LS[v] - ES[v] for v in tasks}
    critical = [v for v in order if reserve[v] == 0]
    return T, ES, EF, LS, LF, reserve, critical, order


# ---- данные -------------------------------------------------------
# Малый тестовый граф (7 вершин, ориентированный): (u, v, w)
TEST_EDGES = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15),
              (2, 3, 11), (2, 5, 2), (3, 4, 6), (5, 4, 9), (4, 6, 3), (3, 6, 12)]
TEST_N = 7
NAMES = "ABCDEFG"       # подписи вершин теста: 0→A, 1→B, ...

# Проект из 12 работ (вариант 13): код → (название, t, предшественники)
PROJECT = {
    "A": ("Сбор требований", 4, []),
    "B": ("Анализ бизнес-процессов", 6, ["A"]),
    "C": ("Проектирование базы данных", 5, ["B"]),
    "D": ("Проектирование интерфейса", 7, ["B"]),
    "E": ("Разработка серверной части", 12, ["C"]),
    "F": ("Разработка клиентской части", 9, ["D"]),
    "G": ("Интеграция модулей", 4, ["E", "F"]),
    "H": ("Подготовка документации", 5, ["C", "D"]),
    "I": ("Тестирование", 7, ["G"]),
    "J": ("Подготовка инфраструктуры", 6, ["C"]),
    "K": ("Обучение пользователей", 3, ["H", "I"]),
    "L": ("Внедрение", 4, ["J", "K"]),
}


def project_tasks():
    return {k: (v[1], list(v[2])) for k, v in PROJECT.items()}


def small_adj():
    adj = [[] for _ in range(TEST_N)]
    for u, v, w in TEST_EDGES:
        adj[u].append((v, w))
    return adj


# ====================================================================
# Измерения
# ====================================================================
def measure(func, *args, repeats=5):
    """Медиана времени по нескольким запускам, сборщик мусора отключён."""
    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
        gc.enable()
    return statistics.median(times), result


def experiment_dijkstra(repeats):
    rows = []
    for n in SIZES:
        m = DEGREE * n
        adj = generate_graph(n, m)
        m_real = count_edges(adj)
        t_arr, (d1, p1) = measure(dijkstra_array, adj, 0, repeats=repeats)
        t_heap, (d2, p2) = measure(dijkstra_heap, adj, 0, repeats=repeats)
        assert d1 == d2, "реализации дали разные расстояния"
        reach = sum(1 for d in d1 if d < INF)
        far = max(range(n), key=lambda v: d1[v] if d1[v] < INF else -1)
        path = restore_path(p2, far)
        # длина восстановленного пути должна совпасть с расстоянием
        wsum = 0
        for a, b in zip(path, path[1:]):
            wsum += min(w for u, w in adj[a] if u == b)
        assert wsum == d1[far]
        rows.append({"n": n, "m": m_real, "t_array": t_arr, "t_heap": t_heap,
                     "ratio": t_arr / t_heap, "reachable": reach,
                     "far": far, "far_dist": d1[far], "path_len": len(path)})
    return rows


def experiment_small():
    adj = small_adj()
    dist, parent, steps = dijkstra_trace(adj, 0)
    d2, _ = dijkstra_heap(adj, 0)
    assert dist == d2
    paths = {NAMES[v]: [NAMES[x] for x in restore_path(parent, v)] for v in range(TEST_N)}
    return {"edges": TEST_EDGES, "dist": dist, "parent": parent, "steps": steps,
            "paths": paths}


def experiment_project():
    tasks = project_tasks()
    T, ES, EF, LS, LF, R, crit, order = critical_path(tasks)
    return {"T": T, "ES": ES, "EF": EF, "LS": LS, "LF": LF, "R": R,
            "critical": crit, "order": order}


def experiment_cycle():
    """К6: исходный график корректен, затем вносится цикл D → … → B."""
    out = {}
    ok = project_tasks()
    out["ok_message"] = "график корректен, длительность %d дн." % critical_path(ok)[0]

    bad = project_tasks()
    bad["B"] = (bad["B"][0], ["A", "I"])        # B зависит от I: B→D→F→G→I→B
    out["cycle_edge"] = "I → B"
    try:
        critical_path(bad)
        out["bad_message"] = "ОШИБКА: цикл не обнаружен"
    except ValueError as e:
        out["bad_message"] = str(e)
    out["cycle"] = find_cycle(bad)

    selfloop = project_tasks()
    selfloop["E"] = (selfloop["E"][0], ["C", "E"])
    try:
        critical_path(selfloop)
        out["self_message"] = "ОШИБКА: цикл не обнаружен"
    except ValueError as e:
        out["self_message"] = str(e)
    out["self_cycle"] = find_cycle(selfloop)

    unknown = project_tasks()
    unknown["L"] = (unknown["L"][0], ["J", "Z"])
    try:
        critical_path(unknown)
        out["unknown_message"] = "ОШИБКА"
    except ValueError as e:
        out["unknown_message"] = str(e)
    return out


def check_correctness(trials=200):
    """Случайные тесты: Дейкстра (оба варианта) против Беллмана — Форда."""
    rng = random.Random(7)
    for _ in range(trials):
        n = rng.randint(2, 30)
        m = rng.randint(n - 1, 4 * n)
        adj = generate_graph(n, m, 20, rng.randint(0, 10 ** 6))
        d1, _ = dijkstra_array(adj, 0)
        d2, _ = dijkstra_heap(adj, 0)
        bf = [INF] * n
        bf[0] = 0
        for _ in range(n - 1):
            for v in range(n):
                if bf[v] < INF:
                    for u, w in adj[v]:
                        if bf[v] + w < bf[u]:
                            bf[u] = bf[v] + w
        assert d1 == d2 == bf
    # случайные сетевые графики: резерв >= 0, критические работы образуют цепь длины T
    for _ in range(trials):
        k = rng.randint(2, 15)
        names = [f"T{i}" for i in range(k)]
        tasks = {}
        for i, nm in enumerate(names):
            preds = [names[j] for j in range(i) if rng.random() < 0.3]
            tasks[nm] = (rng.randint(1, 9), preds)
        T, ES, EF, LS, LF, R, crit, _ = critical_path(tasks)
        assert all(r >= 0 for r in R.values())
        assert any(EF[v] == T for v in crit)
    return True


# ====================================================================
# Вывод
# ====================================================================
def f(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def print_report(res):
    print("=" * 72)
    print("Таблица 1. Время работы алгоритма Дейкстры (медиана), разреженные графы")
    print(f"{'n':>6}{'m':>9}{'массив, с':>12}{'куча, с':>11}{'отношение':>11}")
    for r in res["dijkstra"]:
        print(f"{r['n']:>6}{r['m']:>9}{f(r['t_array']):>12}{f(r['t_heap']):>11}"
              f"{f(r['ratio'], 1):>11}")
    print("\nНаиболее удалённые вершины:")
    for r in res["dijkstra"]:
        print(f"  n={r['n']}: вершина {r['far']}, расстояние {r['far_dist']}, "
              f"вершин на пути {r['path_len']}")

    s = res["small"]
    print("\nТестовый граф (7 вершин), расстояния от A:")
    for v in range(TEST_N):
        print(f"  {NAMES[v]}: d={s['dist'][v]}, путь {' → '.join(s['paths'][NAMES[v]])}")

    p = res["project"]
    print(f"\nДлительность проекта: {p['T']} дн.")
    print(f"{'Работа':>7}{'t':>4}{'ES':>5}{'EF':>5}{'LS':>5}{'LF':>5}{'R':>5}  Крит.")
    for k in sorted(PROJECT):
        print(f"{k:>7}{PROJECT[k][1]:>4}{p['ES'][k]:>5}{p['EF'][k]:>5}{p['LS'][k]:>5}"
              f"{p['LF'][k]:>5}{p['R'][k]:>5}  {'да' if p['R'][k] == 0 else ''}")
    print("Критический путь:", " → ".join(p["critical"]))

    c = res["cycle"]
    print("\nК6. Проверка обнаружения цикла")
    print("  исходные данные:", c["ok_message"])
    print("  внесён цикл через ребро", c["cycle_edge"], "→", c["bad_message"])
    print("  найденный цикл:", " → ".join(c["cycle"]))
    print("  петля E→E:", c["self_message"])
    print("  неизвестный предшественник:", c["unknown_message"])


def save_plots(res, directory):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(directory, exist_ok=True)
    rows = res["dijkstra"]
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ns, [r["t_array"] for r in rows], "o-", label="массив, Θ(n²)")
    axes[0].plot(ns, [r["t_heap"] for r in rows], "s-", label="куча, O(m log n)")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("время, с")
    axes[0].set_title("Время работы")
    axes[0].legend()
    axes[0].grid(alpha=.3)
    axes[1].bar([str(n) for n in ns], [r["ratio"] for r in rows], color="#1f77b4")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("во сколько раз куча быстрее")
    axes[1].set_title("Отношение времён")
    axes[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "dijkstra_times.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", metavar="DIR", help="каталог для графика")
    ap.add_argument("--json", metavar="FILE", help="файл для результатов JSON")
    ap.add_argument("--repeats", type=int, default=5, help="число повторов замера")
    a = ap.parse_args()

    check_correctness()
    print("Проверка корректности на случайных тестах: пройдена")
    res = {"dijkstra": experiment_dijkstra(a.repeats),
           "small": experiment_small(),
           "project": experiment_project(),
           "cycle": experiment_cycle()}
    print_report(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if a.plots:
        save_plots(res, a.plots)


if __name__ == "__main__":
    main()
