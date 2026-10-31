# TC 3 — Cálculo Numérico | Prof. Leduino | Unifesp

## Estrutura da Pasta

```
TC3/
├── main.py                  ← Executa todos os exercícios (Parte 1 + Parte 2)
├── interpolacao.py          ← Parte 1: Interpolação Polinomial (Ex. 1 a 4)
├── minimos_quadrados.py     ← Parte 2: Mínimos Quadrados (Ex. 2 a 6)
└── README.md                ← Este arquivo
```

## Como executar

```bash
cd TC3
python main.py
```

Ou, para rodar cada parte separadamente:

```bash
python interpolacao.py
python minimos_quadrados.py
```

## Resumo dos Exercícios

### Parte 1 — Interpolação Polinomial

| Ex. | Descrição |
|-----|-----------|
| 1 | Polinômio de grau ≤ 3 interpolando 4 pontos (Newton) |
| 2 | Polinômio de grau ≤ 5 com diferenças divididas + estimar f(4.5) |
| 3 | Pressão de vapor da água — estimar P(5), P(45), P(95) |
| 4 | Propriedade C do titânio — estimar C(645), C(795), C(845) |

### Parte 2 — Mínimos Quadrados

| Ex. | Descrição |
|-----|-----------|
| 2 | MQ grau 2 e grau 3 para 7 pontos (+ erro) |
| 3 | MQ grau 2 e grau 3 para f(x)=x²+3x+7 |
| 4 | MQ grau 3 para f(x)=sin(πx) em 5 pontos |
| 5 | MQ grau 2 com estimativa f(6.0) |
| 6 | MQ grau 2 — critério de mínimos quadrados |

---

## Declaração de uso de IAG

Utilizei o assistente de IA Antigravity (Google DeepMind) como apoio no desenvolvimento deste trabalho computacional. A ferramenta me ajudou principalmente na estruturação das funções de interpolação de Newton e de mínimos quadrados, na identificação de erros de indexação nas tabelas de diferenças divididas, e na formatação das saídas dos resultados.

Todas as decisões metodológicas foram minhas: escolhi os algoritmos adequados para cada exercício, analisei os resultados obtidos, comparei com os valores exatos fornecidos pelo enunciado e elaborei as conclusões. Verifiquei a correção dos cálculos e dos códigos gerados antes de incluí-los no trabalho.

---

*Referência: Kharab, Abdelwahab, and Ronald Guenther. An introduction to
numerical methods: a MATLAB® approach. CRC press, 2018.*
