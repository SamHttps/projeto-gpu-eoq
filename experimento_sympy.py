import sympy as sp

def reproduzir_experimento():
    Q, D, P, S, h, theta = sp.symbols('Q D P S h theta', positive=True, real=True)

    C = D * P + (D / Q) * S + (Q / 2) * (h + theta * P)
    print("Função Objetivo C(Q):")
    sp.pprint(C)
    print("-" * 50)

    dC = sp.diff(C, Q)
    print("Primeira Derivada C'(Q):")
    sp.pprint(dC)

    q_star = sp.solve(sp.Eq(dC, 0), Q)[0]
    print("\nFórmula Analítica do Lote Ótimo Q*:")
    sp.pprint(q_star)
    print("-" * 50)

    d2C = sp.diff(dC, Q)
    print("Segunda Derivada C''(Q):")
    sp.pprint(d2C)
    print("-" * 50)

    valores = {
        D: 24000,
        P: 2200.00,
        S: 18500.00,
        h: 45.00,
        theta: 0.25
    }

    q_num = float(q_star.subs(valores))
    c_num = float(C.subs(valores).subs(Q, q_num))
    d2c_num = float(d2C.subs(valores).subs(Q, q_num))

    print("Resultado Numérico (Cenário Base):")
    print(f"-> Lote Ótimo Q* = {q_num:.2f} kits")
    print(f"-> Custo Total Anual Mínimo C(Q*) = R$ {c_num:,.2f}")
    print(f"-> Valor de C''(Q*) = {d2c_num:.6f} (> 0, confirma Mínimo Global)")

if __name__ == "__main__":
    reproduzir_experimento()