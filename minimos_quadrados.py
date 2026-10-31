# -*- coding: utf-8 -*-
"""
TC 3 — Parte 2: Mínimos Quadrados (Least-Squares Approximation)
CN - prof. Leduino - Unifesp
Exercícios do livro Kharab, Abdelwahab, and Ronald Guenther.
An introduction to numerical methods: a MATLAB® approach. CRC press, 2018.

DECLARAÇÃO DE USO DE IAG:
Este trabalho utilizou o assistente Antigravity (Google DeepMind) como apoio
no desenvolvimento do código Python — especificamente na montagem das equações
normais do método de mínimos quadrados, na verificação dos cálculos matriciais
e na estruturação das saídas. Todas as decisões metodológicas, a análise dos
resultados, as comparações e as conclusões foram elaboradas pelos estudantes,
que verificaram a correção dos algoritmos e dos resultados numéricos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Funções auxiliares para Mínimos Quadrados Polinomial
# =============================================================================

def minimos_quadrados_polinomial(x, y, grau):
    """
    Ajusta um polinômio de `grau` dado pelo método de mínimos quadrados.
    Resolve o sistema de equações normais A^T A c = A^T y.
    Retorna os coeficientes c = [c0, c1, ..., c_grau] tal que
    P(x) = c0 + c1*x + c2*x^2 + ... + c_grau*x^grau.
    """
    n = len(x)
    m = grau + 1  # número de coeficientes

    # Monta a matriz de Vandermonde A (n x m)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i, j] = x[i] ** j

    # Equações normais: (A^T A) c = A^T y
    ATA = A.T @ A
    ATy = A.T @ y

    # Resolve o sistema linear
    coefs = np.linalg.solve(ATA, ATy)
    return coefs


def avaliar_polinomio(coefs, x_val):
    """Avalia P(x) = c0 + c1*x + c2*x^2 + ... num ponto x_val."""
    resultado = 0.0
    for j, c in enumerate(coefs):
        resultado += c * (x_val ** j)
    return resultado


def calcular_erro(coefs, x, y):
    """Calcula a norma do resíduo: ||y - P(x)||_2."""
    residuo = np.array([y[i] - avaliar_polinomio(coefs, x[i]) for i in range(len(x))])
    return np.linalg.norm(residuo)


def imprimir_polinomio(coefs, nome="P"):
    """Imprime o polinômio de forma legível."""
    termos = []
    for j, c in enumerate(coefs):
        if j == 0:
            termos.append(f"{c:.4f}")
        elif j == 1:
            termos.append(f"({c:+.4f})x")
        else:
            termos.append(f"({c:+.4f})x^{j}")
    print(f"  {nome}(x) = " + " + ".join(termos).replace("+ (", "("))


# =============================================================================
# Exercício 2 — Mínimos Quadrados grau 2 e grau 3 (7 pontos)
# =============================================================================

def exercicio_p2_2():
    print("=" * 70)
    print("EXERCÍCIO 2 (Parte 2) — Mínimos Quadrados grau 2 e grau 3")
    print("=" * 70)

    x = np.array([1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8])
    y = np.array([5.5, 7.3, 14.1, 26.3, 48.2, 80.1, 111.6])

    print(f"\nDados:")
    print(f"  x : {list(x)}")
    print(f"  y : {list(y)}")

    # (a) Grau 2
    print("\n--- (a) Polinômio de Mínimos Quadrados de GRAU 2 ---")
    coefs2 = minimos_quadrados_polinomial(x, y, 2)
    print(f"\n  Coeficientes:")
    for j, c in enumerate(coefs2):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs2, "P2")
    erro2 = calcular_erro(coefs2, x, y)
    print(f"\n  Erro (norma do resíduo) = {erro2:.6f}")

    # (b) Grau 3
    print("\n--- (b) Polinômio de Mínimos Quadrados de GRAU 3 ---")
    coefs3 = minimos_quadrados_polinomial(x, y, 3)
    print(f"\n  Coeficientes:")
    for j, c in enumerate(coefs3):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs3, "P3")
    erro3 = calcular_erro(coefs3, x, y)
    print(f"\n  Erro (norma do resíduo) = {erro3:.6f}")

    # Comparação
    print(f"\n  Comparação:")
    print(f"  {'x':^6} | {'y real':^10} | {'P2(x)':^12} | {'P3(x)':^12}")
    print("  " + "-" * 47)
    for xi, yi in zip(x, y):
        p2i = avaliar_polinomio(coefs2, xi)
        p3i = avaliar_polinomio(coefs3, xi)
        print(f"  {xi:^6.2f} | {yi:^10.2f} | {p2i:^12.4f} | {p3i:^12.4f}")

    print(f"\n  Conclusão: O polinômio de grau 3 tem erro menor ({erro3:.4f} < {erro2:.4f})")
    print("  e se ajusta melhor ao comportamento exponencial/cúbico dos dados.")

    # Gráfico
    x_plot = np.linspace(0.9, 3.0, 300)
    y2_plot = np.array([avaliar_polinomio(coefs2, xi) for xi in x_plot])
    y3_plot = np.array([avaliar_polinomio(coefs3, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', zorder=5, s=80, label='Dados')
    plt.plot(x_plot, y2_plot, 'b--', linewidth=2, label=f'MQ Grau 2 (erro={erro2:.2f})')
    plt.plot(x_plot, y3_plot, 'g-', linewidth=2, label=f'MQ Grau 3 (erro={erro3:.2f})')
    plt.title("Ex. 2 (P2) — Mínimos Quadrados Grau 2 e 3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p2_ex2_mq.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p2_ex2_mq.png'.")


# =============================================================================
# Exercício 3 — f(x)=x²+3x+7 com ruído; grau 2 e grau 3
# =============================================================================

def exercicio_p2_3():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 3 (Parte 2) — f(x)=x²+3x+7 gerada pelos dados")
    print("=" * 70)

    x = np.array([0.0, 1.0, 2.0, 3.0])
    fx = np.array([7.0, 11.0, 17.0, 25.0])

    print(f"\nFunção geradora: f(x) = x² + 3x + 7")
    print(f"  x    : {list(x)}")
    print(f"  f(x) : {list(fx)}")
    print(f"\n  Nota: 4 pontos com exatamente 4 coeficientes → MQ grau 3 = interpolação exata.")

    # Grau 2
    print("\n--- Polinômio de Mínimos Quadrados de GRAU 2 ---")
    coefs2 = minimos_quadrados_polinomial(x, fx, 2)
    print(f"\n  Coeficientes:")
    for j, c in enumerate(coefs2):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs2, "P2")
    erro2 = calcular_erro(coefs2, x, fx)
    print(f"  Erro (norma do resíduo) = {erro2:.6e}")

    # Grau 3
    print("\n--- Polinômio de Mínimos Quadrados de GRAU 3 ---")
    coefs3 = minimos_quadrados_polinomial(x, fx, 3)
    print(f"\n  Coeficientes:")
    for j, c in enumerate(coefs3):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs3, "P3")
    erro3 = calcular_erro(coefs3, x, fx)
    print(f"  Erro (norma do resíduo) = {erro3:.6e}")

    print(f"\n  Comparação nos 4 pontos:")
    print(f"  {'x':^5} | {'f(x)':^8} | {'P2(x)':^12} | {'P3(x)':^12} | {'f(x) exato':^12}")
    print("  " + "-" * 57)
    for xi, fxi in zip(x, fx):
        p2i = avaliar_polinomio(coefs2, xi)
        p3i = avaliar_polinomio(coefs3, xi)
        f_exato = xi**2 + 3*xi + 7
        print(f"  {xi:^5.0f} | {fxi:^8.1f} | {p2i:^12.4f} | {p3i:^12.4f} | {f_exato:^12.1f}")

    print(f"\n  Análise:")
    print("  Como f(x) = x² + 3x + 7 é um polinômio de grau 2, os dados são EXATOS.")
    print("  O MQ grau 2 recupera perfeitamente os coeficientes (c0=7, c1=3, c2=1).")
    print("  O MQ grau 3 é equivalente à interpolação e também recupera os dados,")
    print("  mas com coeficiente c3 ≈ 0 (numericamente).")

    # Gráfico
    x_plot = np.linspace(-0.2, 3.2, 300)
    f_exata = x_plot**2 + 3*x_plot + 7
    y2_plot = np.array([avaliar_polinomio(coefs2, xi) for xi in x_plot])
    y3_plot = np.array([avaliar_polinomio(coefs3, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_exata, 'k--', linewidth=1.5, label='f(x) = x²+3x+7 (exato)', alpha=0.6)
    plt.scatter(x, fx, color='red', zorder=5, s=80, label='Dados')
    plt.plot(x_plot, y2_plot, 'b-', linewidth=2, label=f'MQ Grau 2')
    plt.plot(x_plot, y3_plot, 'g:', linewidth=2.5, label=f'MQ Grau 3 (interpola)')
    plt.title("Ex. 3 (P2) — MQ de f(x)=x²+3x+7")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p2_ex3_mq.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p2_ex3_mq.png'.")


# =============================================================================
# Exercício 4 — f(x)=sin(πx) nos pontos x=-1,-0.5,0,0.5,1; grau 3
# =============================================================================

def exercicio_p2_4():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 4 (Parte 2) — Ajuste de grau 3 para f(x)=sin(πx)")
    print("=" * 70)

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    fx = np.sin(np.pi * x)

    print(f"\nFunção: f(x) = sin(πx)")
    print(f"  Pontos: x = [-1, -0.5, 0, 0.5, 1]")
    print(f"  f(x)  : {[f'{v:.6f}' for v in fx]}")

    coefs = minimos_quadrados_polinomial(x, fx, 3)
    print(f"\n  Coeficientes do polinômio de grau 3 (MQ):")
    for j, c in enumerate(coefs):
        print(f"    c_{j} = {c:.8f}")
    imprimir_polinomio(coefs, "P3")

    erro = calcular_erro(coefs, x, fx)
    print(f"\n  Erro (norma do resíduo) = {erro:.6e}")

    print(f"\n  Avaliação nos pontos dados:")
    print(f"  {'x':^7} | {'sin(πx)':^12} | {'P3(x)':^12} | {'Resíduo':^12}")
    print("  " + "-" * 50)
    for xi, fxi in zip(x, fx):
        p3i = avaliar_polinomio(coefs, xi)
        print(f"  {xi:^7.2f} | {fxi:^12.6f} | {p3i:^12.6f} | {abs(p3i-fxi):^12.2e}")

    print(f"\n  Análise:")
    print("  sin(πx) é ímpar → seus coeficientes de Taylor são todos ímpares.")
    print("  O MQ grau 3 (polinômio ímpar) deve ter c0≈0 e c2≈0 (por simetria),")
    print("  resultando em P3(x) ≈ c1*x + c3*x³.")

    # Gráfico
    x_plot = np.linspace(-1.1, 1.1, 400)
    f_exata = np.sin(np.pi * x_plot)
    y3_plot = np.array([avaliar_polinomio(coefs, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, f_exata, 'k--', linewidth=1.5, label='sin(πx) exato')
    plt.scatter(x, fx, color='red', zorder=5, s=80, label='Pontos usados no ajuste')
    plt.plot(x_plot, y3_plot, 'b-', linewidth=2, label='MQ Grau 3')
    plt.title("Ex. 4 (P2) — MQ Grau 3 para sin(πx)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p2_ex4_sin.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p2_ex4_sin.png'.")


# =============================================================================
# Exercício 5 — MQ grau 2 com 6 pontos; estimar f(6.0)
# =============================================================================

def exercicio_p2_5():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 5 (Parte 2) — MQ Grau 2 com estimativa f(6.0)")
    print("=" * 70)

    x = np.array([1.2, 3.0, 4.5, 5.6, 6.9, 8.1])
    y = np.array([2.3, 3.3, 5.6, 15.4, 38.2, 59.1])

    print(f"\nDados:")
    print(f"  x : {list(x)}")
    print(f"  y : {list(y)}")

    coefs2 = minimos_quadrados_polinomial(x, y, 2)
    print(f"\n  Coeficientes do polinômio de grau 2 (MQ):")
    for j, c in enumerate(coefs2):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs2, "P2")

    erro = calcular_erro(coefs2, x, y)
    print(f"\n  Erro (norma do resíduo) = {erro:.6f}")

    x_estimar = 6.0
    y_estimado = avaliar_polinomio(coefs2, x_estimar)
    print(f"\n  Estimativa em x = {x_estimar}:")
    print(f"    P2({x_estimar}) = {y_estimado:.4f}")

    print(f"\n  Comparação nos pontos dados:")
    print(f"  {'x':^6} | {'y real':^10} | {'P2(x)':^12} | {'Resíduo':^12}")
    print("  " + "-" * 47)
    for xi, yi in zip(x, y):
        p2i = avaliar_polinomio(coefs2, xi)
        print(f"  {xi:^6.2f} | {yi:^10.2f} | {p2i:^12.4f} | {abs(p2i-yi):^12.4f}")

    # Gráfico
    x_plot = np.linspace(1.0, 9.0, 300)
    y2_plot = np.array([avaliar_polinomio(coefs2, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='red', zorder=5, s=80, label='Dados')
    plt.plot(x_plot, y2_plot, 'b-', linewidth=2, label=f'MQ Grau 2')
    plt.scatter([x_estimar], [y_estimado], color='green', zorder=6, s=100,
                marker='*', label=f'P2(6.0) ≈ {y_estimado:.4f}')
    plt.title("Ex. 5 (P2) — MQ Grau 2 e estimativa f(6.0)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p2_ex5_mq.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p2_ex5_mq.png'.")


# =============================================================================
# Exercício 6 — MQ grau 2 com 6 pontos (critério de mínimos quadrados)
# =============================================================================

def exercicio_p2_6():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 6 (Parte 2) — MQ Grau 2 (critério de mínimos quadrados)")
    print("=" * 70)

    x = np.array([1.0, 2.3, 3.8, 4.9, 6.1, 7.2])
    fx = np.array([0.1, 1.5, 7.4, 15.2, 26.2, 27.9])

    print(f"\nDados:")
    print(f"  x    : {list(x)}")
    print(f"  f(x) : {list(fx)}")

    coefs2 = minimos_quadrados_polinomial(x, fx, 2)
    print(f"\n  Coeficientes do polinômio de grau 2 (MQ):")
    for j, c in enumerate(coefs2):
        print(f"    c_{j} = {c:.6f}")
    imprimir_polinomio(coefs2, "P2")

    erro = calcular_erro(coefs2, x, fx)
    print(f"\n  Erro (norma do resíduo) = {erro:.6f}")

    # Equações normais explicitadas
    n = len(x)
    print(f"\n  Equações Normais (sistema A^T A c = A^T y):")
    S = {}
    for k in range(5):
        S[f's{k}'] = float(np.sum(x**k))
    Sy0 = float(np.sum(fx))
    Sy1 = float(np.sum(x * fx))
    Sy2 = float(np.sum(x**2 * fx))
    print(f"  n={n}, Σx={S['s1']:.4f}, Σx²={S['s2']:.4f}, Σx³={S['s3']:.4f}, Σx⁴={S['s4']:.4f}")
    print(f"  Σy={Sy0:.4f}, Σxy={Sy1:.4f}, Σx²y={Sy2:.4f}")
    print(f"\n  Sistema:")
    print(f"  | {n:>8.4f}  {S['s1']:>10.4f}  {S['s2']:>10.4f} | | c0 |   | {Sy0:>10.4f} |")
    print(f"  | {S['s1']:>8.4f}  {S['s2']:>10.4f}  {S['s3']:>10.4f} | | c1 | = | {Sy1:>10.4f} |")
    print(f"  | {S['s2']:>8.4f}  {S['s3']:>10.4f}  {S['s4']:>10.4f} | | c2 |   | {Sy2:>10.4f} |")

    print(f"\n  Comparação nos pontos dados:")
    print(f"  {'x':^6} | {'f(x)':^10} | {'P2(x)':^12} | {'Resíduo':^12}")
    print("  " + "-" * 47)
    for xi, fxi in zip(x, fx):
        p2i = avaliar_polinomio(coefs2, xi)
        print(f"  {xi:^6.2f} | {fxi:^10.2f} | {p2i:^12.4f} | {abs(p2i-fxi):^12.4f}")

    # Gráfico
    x_plot = np.linspace(0.5, 8.0, 300)
    y2_plot = np.array([avaliar_polinomio(coefs2, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.scatter(x, fx, color='red', zorder=5, s=80, label='Dados')
    plt.plot(x_plot, y2_plot, 'b-', linewidth=2, label='MQ Grau 2')
    plt.title("Ex. 6 (P2) — MQ Grau 2")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p2_ex6_mq.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p2_ex6_mq.png'.")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    exercicio_p2_2()
    exercicio_p2_3()
    exercicio_p2_4()
    exercicio_p2_5()
    exercicio_p2_6()

    print("\n" + "=" * 70)
    print("TC 3 Parte 2 — Mínimos Quadrados concluído.")
    print("=" * 70)
