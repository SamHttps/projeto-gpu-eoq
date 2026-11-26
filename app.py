import logging
import sqlite3
import csv
import io
from datetime import datetime
import numpy as np
import sympy as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field, model_validator

# ==========================================
# CONFIGURAÇÃO DE LOGS E APLICAÇÃO
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("gpu_eoq_system.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AmazonasGPUSupplyChain")

app = FastAPI(
    title="API de Otimização Logística - Amazonas GPU Tech",
    description="Sistema de Apoio à Decisão baseado em Cálculo Diferencial (EOQ com Obsolescência) usando SymPy.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "simulacoes_gpu.db"

# ==========================================
# PERSISTÊNCIA EM BANCO DE DADOS (SQLITE)
# ==========================================
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT NOT NULL,
                cenario_nome TEXT NOT NULL,
                demanda_anual REAL NOT NULL,
                custo_unitario REAL NOT NULL,
                custo_pedido REAL NOT NULL,
                custo_armazenagem REAL NOT NULL,
                taxa_obsolescencia REAL NOT NULL,
                q_min REAL NOT NULL,
                q_max REAL NOT NULL,
                q_critico_irrestrito REAL NOT NULL,
                q_otimo_recomendado REAL NOT NULL,
                custo_total_otimo REAL NOT NULL,
                custo_logistico_otimo REAL NOT NULL,
                pedidos_por_ano REAL NOT NULL,
                status_fronteira TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Banco de dados SQLite inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}")

init_db()

# ==========================================
# MODELOS DE ENTRADA E VALIDAÇÃO (PYDANTIC)
# ==========================================
class SimulacaoInput(BaseModel):
    cenario_nome: str = Field(default="Lote Kits RTX-5080 (GPU + GDDR7)", min_length=3, max_length=100)
    demanda_anual: float = Field(..., gt=0)
    custo_unitario: float = Field(..., gt=0)
    custo_pedido: float = Field(..., gt=0)
    custo_armazenagem: float = Field(..., ge=0)
    taxa_obsolescencia: float = Field(..., ge=0, le=5.0)
    q_min: float = Field(default=50.0, gt=0)
    q_max: float = Field(default=5000.0, gt=0)

    @model_validator(mode="after")
    def validar_dominio(self):
        if self.q_min >= self.q_max:
            raise ValueError("O limite inferior (Q_min) deve ser estritamente menor que o limite superior (Q_max).")
        if (self.custo_armazenagem + self.taxa_obsolescencia * self.custo_unitario) <= 0:
            raise ValueError("O custo total de manutenção de estoque (h + theta*P) deve ser maior que zero.")
        return self

# ==========================================
# MOTOR MATEMÁTICO SIMBÓLICO (SYMPY)
# ==========================================
def resolver_modelo_sympy(dados: SimulacaoInput):
    Q, D, P, S, h, theta = sp.symbols("Q D P S h theta", positive=True, real=True)

    # Função Objetivo: Custo Total Anual C(Q)
    C_expr = D * P + (D / Q) * S + (Q / 2) * (h + theta * P)

    # Derivadas simbólicas
    dC_dQ = sp.diff(C_expr, Q)
    d2C_dQ2 = sp.diff(dC_dQ, Q)

    # Ponto crítico irrestrito: C'(Q) = 0
    pontos_criticos = sp.solve(sp.Eq(dC_dQ, 0), Q)
    q_critico_sym = pontos_criticos[0]

    subs_dict = {
        D: dados.demanda_anual,
        P: dados.custo_unitario,
        S: dados.custo_pedido,
        h: dados.custo_armazenagem,
        theta: dados.taxa_obsolescencia
    }

    C_num = C_expr.subs(subs_dict)
    dC_num = dC_dQ.subs(subs_dict)
    d2C_num = d2C_dQ2.subs(subs_dict)

    q_critico = float(q_critico_sym.subs(subs_dict))
    segunda_derivada_no_critico = float(d2C_num.subs(Q, q_critico))

    # Avaliação de Domínio Restrito [Q_min, Q_max] e Fronteiras
    custo_q_min = float(C_num.subs(Q, dados.q_min))
    custo_q_max = float(C_num.subs(Q, dados.q_max))

    if dados.q_min <= q_critico <= dados.q_max:
        q_otimo = q_critico
        status_fronteira = "Ponto Interior (Ótimo Global Irrestrito dentro da capacidade)"
    elif q_critico < dados.q_min:
        q_otimo = dados.q_min
        status_fronteira = f"Fronteira Inferior Atingida (Q*={q_critico:.1f} < MOQ={dados.q_min:.0f})"
    else:
        q_otimo = dados.q_max
        status_fronteira = f"Fronteira Superior Atingida (Q*={q_critico:.1f} > Cofre={dados.q_max:.0f})"

    custo_total_otimo = float(C_num.subs(Q, q_otimo))
    custo_aquisicao = dados.demanda_anual * dados.custo_unitario
    custo_logistico_otimo = custo_total_otimo - custo_aquisicao
    pedidos_ano = dados.demanda_anual / q_otimo
    ciclo_dias = 365.0 / pedidos_ano

    q_plot_min = max(10.0, min(dados.q_min * 0.7, q_otimo * 0.4))
    q_plot_max = max(dados.q_max * 1.15, q_otimo * 1.8)
    q_vals = np.linspace(q_plot_min, q_plot_max, 60)
    f_lambdified = sp.lambdify(Q, C_num, "numpy")
    c_vals = f_lambdified(q_vals)

    sensibilidade = []
    for fator in [0.7, 0.85, 1.0, 1.15, 1.3]:
        theta_var = dados.taxa_obsolescencia * fator
        H_var = dados.custo_armazenagem + theta_var * dados.custo_unitario
        q_sens = float(np.sqrt((2 * dados.demanda_anual * dados.custo_pedido) / H_var))
        q_sens_viavel = min(max(q_sens, dados.q_min), dados.q_max)
        c_sens = float(C_expr.subs({
            D: dados.demanda_anual, P: dados.custo_unitario, S: dados.custo_pedido,
            h: dados.custo_armazenagem, theta: theta_var, Q: q_sens_viavel
        }))
        sensibilidade.append({
            "variacao_theta_pct": round((fator - 1.0) * 100, 1),
            "theta_aplicado": round(theta_var * 100, 2),
            "q_otimo": round(q_sens_viavel, 2),
            "custo_total": round(c_sens, 2)
        })

    q_mensal = min(max(dados.demanda_anual / 12.0, dados.q_min), dados.q_max)
    custo_mensal = float(C_num.subs(Q, q_mensal))
    economia_anual = custo_mensal - custo_total_otimo

    return {
        "equacoes_latex": {
            "funcao_objetivo": sp.latex(C_expr),
            "primeira_derivada": sp.latex(dC_dQ),
            "segunda_derivada": sp.latex(d2C_dQ2),
            "formula_ponto_critico": sp.latex(q_critico_sym),
            "funcao_instanciada": sp.latex(sp.N(C_num, 4)),
            "primeira_derivada_instanciada": sp.latex(sp.N(dC_num, 4)),
            "segunda_derivada_instanciada": sp.latex(sp.N(d2C_num, 4))
        },
        "resultados": {
            "q_critico_irrestrito": round(q_critico, 2),
            "q_otimo_recomendado": round(q_otimo, 2),
            "q_otimo_inteiro": int(round(q_otimo)),
            "segunda_derivada_valor": round(segunda_derivada_no_critico, 6),
            "concavidade": "Estritamente Convexa (C''(Q) > 0) -> Mínimo Global Garantido",
            "status_fronteira": status_fronteira,
            "custo_q_min": round(custo_q_min, 2),
            "custo_q_max": round(custo_q_max, 2),
            "custo_total_otimo": round(custo_total_otimo, 2),
            "custo_logistico_otimo": round(custo_logistico_otimo, 2),
            "custo_aquisicao_anual": round(custo_aquisicao, 2),
            "pedidos_por_ano": round(pedidos_ano, 2),
            "intervalo_dias": round(ciclo_dias, 1),
            "comparacao_pedido_mensal": {
                "q_mensal": round(q_mensal, 2),
                "custo_mensal": round(custo_mensal, 2),
                "economia_anual_estimada": round(economia_anual, 2)
            }
        },
        "grafico": {
            "q_values": [round(float(x), 2) for x in q_vals],
            "c_values": [round(float(y), 2) for y in c_vals]
        },
        "sensibilidade": sensibilidade
    }

# ==========================================
# ENDPOINTS DA API REST
# ==========================================
@app.get("/")
def servir_frontend():
    return FileResponse("index.html")

@app.post("/api/otimizar")
def otimizar_estoque(dados: SimulacaoInput):
    try:
        logger.info(f"Iniciando otimização para cenário: {dados.cenario_nome}")
        analise = resolver_modelo_sympy(dados)
        res = analise["resultados"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO simulacoes (
                data_hora, cenario_nome, demanda_anual, custo_unitario, custo_pedido,
                custo_armazenagem, taxa_obsolescencia, q_min, q_max,
                q_critico_irrestrito, q_otimo_recomendado, custo_total_otimo,
                custo_logistico_otimo, pedidos_por_ano, status_fronteira
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agora, dados.cenario_nome, dados.demanda_anual, dados.custo_unitario,
            dados.custo_pedido, dados.custo_armazenagem, dados.taxa_obsolescencia,
            dados.q_min, dados.q_max, res["q_critico_irrestrito"],
            res["q_otimo_recomendado"], res["custo_total_otimo"],
            res["custo_logistico_otimo"], res["pedidos_por_ano"], res["status_fronteira"]
        ))
        sim_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Simulação ID={sim_id} concluída. Q*={res['q_otimo_recomendado']}")
        return {"id": sim_id, "data_hora": agora, "entrada": dados.model_dump(), **analise}

    except ValueError as ve:
        logger.warning(f"Erro de validação: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro interno na otimização: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar modelagem simbólica.")

@app.get("/api/historico")
def listar_historico(limit: int = Query(default=20, ge=1, le=100)):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM simulacoes ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Erro ao buscar histórico: {e}")
        raise HTTPException(status_code=500, detail="Erro ao acessar banco de dados.")

@app.delete("/api/historico/{sim_id}")
def deletar_simulacao(sim_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM simulacoes WHERE id = ?", (sim_id,))
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Simulação não encontrada.")
        conn.commit()
        conn.close()
        logger.info(f"Simulação ID={sim_id} removida.")
        return {"mensagem": "Simulação removida com sucesso."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao deletar simulação: {e}")
        raise HTTPException(status_code=500, detail="Erro ao deletar registro.")

@app.get("/api/exportar-csv")
def exportar_csv():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM simulacoes ORDER BY id DESC")
        rows = cursor.fetchall()
        headers = [description[0] for description in cursor.description]
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=historico_otimizacao_gpu.csv"}
        )
    except Exception as e:
        logger.error(f"Erro ao exportar CSV: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerar arquivo CSV.")