# -*- coding: utf-8 -*-
"""
TC 3 — Trabalho Computacional 3
Cálculo Numérico — prof. Leduino — Unifesp

Tópicos:
  • Parte 1: Interpolação Polinomial (diferenças divididas de Newton)
  • Parte 2: Aproximação por Mínimos Quadrados

Execute este arquivo para rodar todos os exercícios das duas partes.
Os gráficos serão salvos no diretório corrente (TC3/).
"""

import os
import sys

# Garante saída UTF-8 no Windows (evita UnicodeEncodeError com acentos)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Garante que o diretório de trabalho é o da pasta TC3
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Importa e executa a Parte 1 — Interpolação Polinomial
# =============================================================================
from interpolacao import (
    exercicio_p1_1,
    exercicio_p1_2,
    exercicio_p1_3,
    exercicio_p1_4,
)

# =============================================================================
# Importa e executa a Parte 2 — Mínimos Quadrados
# =============================================================================
from minimos_quadrados import (
    exercicio_p2_2,
    exercicio_p2_3,
    exercicio_p2_4,
    exercicio_p2_5,
    exercicio_p2_6,
)


def separador(titulo):
    print("\n")
    print("#" * 70)
    print(f"  {titulo}")
    print("#" * 70)


if __name__ == '__main__':
    separador("TC 3 — PARTE 1: INTERPOLAÇÃO POLINOMIAL")
    exercicio_p1_1()
    exercicio_p1_2()
    exercicio_p1_3()
    exercicio_p1_4()

    separador("TC 3 — PARTE 2: MÍNIMOS QUADRADOS")
    exercicio_p2_2()
    exercicio_p2_3()
    exercicio_p2_4()
    exercicio_p2_5()
    exercicio_p2_6()

    print("\n\n" + "=" * 70)
    print("  TC 3 — Todos os exercícios concluídos com sucesso!")
    print("  Gráficos gerados nesta pasta (TC3/).")
    print("=" * 70)
