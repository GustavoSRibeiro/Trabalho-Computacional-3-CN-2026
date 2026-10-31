# -*- coding: utf-8 -*-
"""
TC 3 — Parte 1: Interpolação Polinomial
CN - prof. Leduino - Unifesp
Exercícios do livro Kharab, Abdelwahab, and Ronald Guenther.
An introduction to numerical methods: a MATLAB® approach. CRC press, 2018.

DECLARAÇÃO DE USO DE IAG:
Este trabalho utilizou o assistente Antigravity (Google DeepMind) como apoio
no desenvolvimento do código Python — especificamente na estruturação das
funções de interpolação de Newton e Lagrange, identificação de erros de
indexação e formatação das tabelas de saída. Todas as decisões metodológicas,
a análise dos resultados, as comparações com valores exatos e as conclusões
foram elaboradas pelos estudantes, que verificaram a correção dos algoritmos
e dos resultados numéricos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Funções auxiliares de Interpolação
# =============================================================================

def diferencias_divididas(x, y):
    """
    Constrói a tabela de diferenças divididas de Newton.
    Retorna os coeficientes [f[x0], f[x0,x1], f[x0,x1,x2], ...].
    """
    n = len(x)
    # Tabela triangular; dd[i][j] = f[x_i, ..., x_{i+j}]
    dd = [list(y)]
    for j in range(1, n):
        coluna = []
        for i in range(n - j):
            val = (dd[j-1][i+1] - dd[j-1][i]) / (x[i+j] - x[i])
            coluna.append(val)
        dd.append(coluna)
    # Coeficientes são os primeiros elementos de cada coluna
    coefs = [dd[j][0] for j in range(n)]
    return coefs


def avaliar_newton(coefs, x_nos, x_val):
    """
    Avalia o polinômio de Newton com diferenças divididas num ponto x_val
    usando o método de Horner generalizado.
    """
    n = len(coefs)
    resultado = coefs[n - 1]
    for i in range(n - 2, -1, -1):
        resultado = resultado * (x_val - x_nos[i]) + coefs[i]
    return resultado


def imprimir_tabela_dd(x, y):
    """Imprime a tabela de diferenças divididas formatada."""
    n = len(x)
    dd = [list(y)]
    for j in range(1, n):
        coluna = []
        for i in range(n - j):
            val = (dd[j-1][i+1] - dd[j-1][i]) / (x[i+j] - x[i])
            coluna.append(val)
        dd.append(coluna)

    print(f"\n  Tabela de Diferenças Divididas:")
    # Cabeçalho
    header = f"  {'i':^4} | {'x_i':^8} | {'f[xi]':^12}"
    for j in range(1, n):
        header += f" | {'f['+','.join([f'x{k}' for k in range(j+1)])+']':^14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i in range(n):
        linha = f"  {i:^4} | {x[i]:^8.4f} | {dd[0][i]:^12.6f}"
        for j in range(1, n - i):
            if i < len(dd[j]):
                linha += f" | {dd[j][i]:^14.6f}"
            else:
                linha += f" | {'':^14}"
        print(linha)


# =============================================================================
# Exercício 1 — Polinômio de grau ≤ 3 por diferenças divididas de Newton
# =============================================================================

def exercicio_p1_1():
    print("=" * 70)
    print("EXERCÍCIO 1 (Parte 1) — Polinômio de grau ≤ 3 interpolando os dados")
    print("=" * 70)

    x = np.array([1.2, 2.1, 3.0, 3.6])
    y = np.array([0.7, 8.1, 27.7, 45.1])

    print(f"\nDados:")
    print(f"  x : {list(x)}")
    print(f"  y : {list(y)}")

    coefs = diferencias_divididas(x, y)

    imprimir_tabela_dd(x, y)

    print(f"\n  Coeficientes (diferenças divididas):")
    for i, c in enumerate(coefs):
        print(f"    a_{i} = {c:.6f}")

    print(f"\n  Polinômio de Newton:")
    print(f"  P(x) = {coefs[0]:.4f} + {coefs[1]:.4f}*(x-{x[0]}) "
          f"+ {coefs[2]:.4f}*(x-{x[0]})*(x-{x[1]}) "
          f"+ {coefs[3]:.4f}*(x-{x[0]})*(x-{x[1]})*(x-{x[2]})")

    # Verificação nos nós
    print(f"\n  Verificação nos nós de interpolação:")
    print(f"  {'x':^8} | {'y_real':^12} | {'P(x)':^12} | {'Erro':^14}")
    print("  " + "-" * 52)
    for xi, yi in zip(x, y):
        px = avaliar_newton(coefs, x, xi)
        print(f"  {xi:^8.2f} | {yi:^12.4f} | {px:^12.6f} | {abs(px-yi):^14.2e}")

    # Gráfico
    x_plot = np.linspace(1.0, 3.8, 300)
    y_plot = np.array([avaliar_newton(coefs, x, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio de Newton (grau 3)')
    plt.scatter(x, y, color='red', zorder=5, s=80, label='Dados (nós)')
    plt.title("Ex. 1 (P1) — Interpolação de Newton (grau ≤ 3)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p1_ex1_interpolacao.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p1_ex1_interpolacao.png'.")


# =============================================================================
# Exercício 2 — Polinômio de grau ≤ 5 via diferenças divididas + f(4.5)
# =============================================================================

def exercicio_p1_2():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 2 (Parte 1) — Polinômio de grau ≤ 5 (diferenças divididas)")
    print("=" * 70)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fx = np.array([14.5, 19.5, 30.5, 53.5, 94.5, 159.5])

    print(f"\nDados:")
    print(f"  x    : {list(x)}")
    print(f"  f(x) : {list(fx)}")

    coefs = diferencias_divididas(x, fx)

    imprimir_tabela_dd(x, fx)

    print(f"\n  Coeficientes do polinômio de Newton (grau ≤ 5):")
    for i, c in enumerate(coefs):
        print(f"    a_{i} = {c:.6f}")

    x_estimar = 4.5
    p_estimado = avaliar_newton(coefs, x, x_estimar)
    exato = 71.375
    erro_abs = abs(p_estimado - exato)
    erro_rel = erro_abs / abs(exato) * 100

    print(f"\n  Estimativa em x = {x_estimar}:")
    print(f"    P({x_estimar})   = {p_estimado:.6f}")
    print(f"    Valor exato   = {exato:.6f}")
    print(f"    Erro absoluto = {erro_abs:.6e}")
    print(f"    Erro relativo = {erro_rel:.4f}%")

    # Gráfico
    x_plot = np.linspace(0.8, 6.2, 300)
    y_plot = np.array([avaliar_newton(coefs, x, xi) for xi in x_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio de Newton (grau ≤ 5)')
    plt.scatter(x, fx, color='red', zorder=5, s=80, label='Dados (nós)')
    plt.scatter([x_estimar], [p_estimado], color='green', zorder=6, s=100,
                marker='*', label=f'P(4.5) ≈ {p_estimado:.4f}')
    plt.scatter([x_estimar], [exato], color='orange', zorder=6, s=100,
                marker='D', label=f'f(4.5) exato = {exato}')
    plt.title("Ex. 2 (P1) — Interpolação de Newton (grau ≤ 5)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p1_ex2_interpolacao.png", dpi=110)
    plt.close()
    print(f"  Gráfico salvo em 'p1_ex2_interpolacao.png'.")


# =============================================================================
# Exercício 3 — Pressão de vapor da água (interpolação)
# =============================================================================

def exercicio_p1_3():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 3 (Parte 1) — Pressão de vapor da água P(T)")
    print("=" * 70)

    T = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0])
    P = np.array([0.0061, 0.0123, 0.0234, 0.0424, 0.0738, 0.1992, 0.4736, 1.0133])

    print(f"\nDados:")
    print(f"  T (°C) : {list(T)}")
    print(f"  P (bar): {list(P)}")
    print(f"\n  Nota: 8 pontos → polinômio de grau ≤ 7.")

    coefs = diferencias_divididas(T, P)

    print(f"\n  Coeficientes do polinômio de Newton (grau ≤ 7):")
    for i, c in enumerate(coefs):
        print(f"    a_{i} = {c:.8e}")

    pontos_estimar = [5.0, 45.0, 95.0]
    exatos = [0.008721, 0.095848, 0.84528]

    print(f"\n  Estimativas e comparação com valores exatos:")
    print(f"  {'T':^6} | {'P estimado':^14} | {'P exato':^14} | {'Erro abs.':^14} | {'Erro rel.':^12}")
    print("  " + "-" * 68)
    for T_val, P_ex in zip(pontos_estimar, exatos):
        P_est = avaliar_newton(coefs, T, T_val)
        err_abs = abs(P_est - P_ex)
        err_rel = err_abs / abs(P_ex) * 100
        print(f"  {T_val:^6.0f} | {P_est:^14.6f} | {P_ex:^14.6f} | {err_abs:^14.6e} | {err_rel:^12.4f}%")

    # Gráfico
    T_plot = np.linspace(0, 100, 500)
    P_plot = np.array([avaliar_newton(coefs, T, ti) for ti in T_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(T_plot, P_plot, 'b-', linewidth=2, label='Polinômio interpolador (grau ≤ 7)')
    plt.scatter(T, P, color='red', zorder=5, s=80, label='Dados tabelados')
    for T_val, P_ex in zip(pontos_estimar, exatos):
        P_est = avaliar_newton(coefs, T, T_val)
        plt.scatter([T_val], [P_est], zorder=6, s=80, marker='*',
                    label=f'P({T_val:.0f}) ≈ {P_est:.4f}')
    plt.title("Ex. 3 (P1) — Pressão de Vapor da Água vs Temperatura")
    plt.xlabel("Temperatura T (°C)")
    plt.ylabel("Pressão P (bar)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p1_ex3_vapor.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p1_ex3_vapor.png'.")

    print(f"\n  Análise:")
    print("  O polinômio de grau 7 interpola exatamente os 8 pontos tabelados.")
    print("  Os erros nas estimativas são pequenos (<1%), pois as temperaturas")
    print("  pedidas (5, 45, 95°C) estão dentro do intervalo de interpolação.")


# =============================================================================
# Exercício 4 — Propriedade C do Titânio em função de T
# =============================================================================

def exercicio_p1_4():
    print("\n" + "=" * 70)
    print("EXERCÍCIO 4 (Parte 1) — Propriedade C do Titânio vs Temperatura T")
    print("=" * 70)

    T = np.array([605.0, 685.0, 725.0, 765.0, 825.0, 855.0, 875.0])
    C = np.array([0.622, 0.655, 0.668, 0.679, 0.730, 0.907, 1.336])

    print(f"\nDados:")
    print(f"  T : {list(T)}")
    print(f"  C : {list(C)}")
    print(f"\n  7 pontos → polinômio de grau ≤ 6.")

    coefs = diferencias_divididas(T, C)

    print(f"\n  Coeficientes do polinômio de Newton (grau ≤ 6):")
    for i, c in enumerate(coefs):
        print(f"    a_{i} = {c:.8e}")

    pontos_estimar = [645.0, 795.0, 845.0]
    exatos = [0.639, 0.694, 0.812]

    print(f"\n  Estimativas e comparação com valores conhecidos:")
    print(f"  {'T':^6} | {'C estimado':^14} | {'C exato':^14} | {'Erro abs.':^14} | {'Erro rel.':^12}")
    print("  " + "-" * 68)
    for T_val, C_ex in zip(pontos_estimar, exatos):
        C_est = avaliar_newton(coefs, T, T_val)
        err_abs = abs(C_est - C_ex)
        err_rel = err_abs / abs(C_ex) * 100
        print(f"  {T_val:^6.0f} | {C_est:^14.6f} | {C_ex:^14.6f} | {err_abs:^14.6e} | {err_rel:^12.4f}%")

    # Gráfico
    T_plot = np.linspace(600, 880, 500)
    C_plot = np.array([avaliar_newton(coefs, T, ti) for ti in T_plot])

    plt.figure(figsize=(8, 5))
    plt.plot(T_plot, C_plot, 'b-', linewidth=2, label='Polinômio interpolador (grau ≤ 6)')
    plt.scatter(T, C, color='red', zorder=5, s=80, label='Dados do titânio')
    for T_val, C_ex in zip(pontos_estimar, exatos):
        C_est = avaliar_newton(coefs, T, T_val)
        plt.scatter([T_val], [C_est], zorder=6, s=80, marker='*',
                    label=f'C({T_val:.0f}) ≈ {C_est:.4f}')
    plt.title("Ex. 4 (P1) — Propriedade C do Titânio vs Temperatura")
    plt.xlabel("Temperatura T (°C)")
    plt.ylabel("Propriedade C")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig("p1_ex4_titanio.png", dpi=110)
    plt.close()
    print(f"\n  Gráfico salvo em 'p1_ex4_titanio.png'.")

    print(f"\n  Análise:")
    print("  O erro em C(845) é ligeiramente maior (~1%) pois o dado C=1.336 em T=875")
    print("  é um salto abrupto (transição de fase?) que gera oscilação de Runge em")
    print("  T próximos de 875. Para C(645) e C(795) os erros são bem menores pois")
    print("  estão em regiões de comportamento suave da curva.")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    exercicio_p1_1()
    exercicio_p1_2()
    exercicio_p1_3()
    exercicio_p1_4()

    print("\n" + "=" * 70)
    print("TC 3 Parte 1 — Interpolação Polinomial concluído.")
    print("=" * 70)
