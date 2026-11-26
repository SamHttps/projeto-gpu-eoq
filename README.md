# Amazonas GPU Tech — Otimizador de Lote Econômico de Compra (EOQ) com Obsolescência Tecnológica

Sistema full stack de apoio à decisão logística desenvolvido para a disciplina **Resolução Diferencial de Problemas** (Curso de Ciência da Computação — CESUPA, Prof. Pedro Girotto). A aplicação utiliza **Cálculo Diferencial** e computação simbólica (**SymPy**) para determinar o tamanho ótimo de lotes de importação de semicondutores (GPUs e memórias GDDR), minimizando custos anuais de pedido, armazenagem climatizada e perda por obsolescência tecnológica.

---

## 1. Contexto Científico e Recorte do Problema

A modelagem matemática baseia-se na literatura moderna de gestão de cadeias de suprimentos de alta tecnologia:

- **Artigo Principal:** _Economic Order Quantity: A State-of-the-Art in the Era of Uncertain Supply Chains_ — _Sustainability_ (MDPI), v. 16, n. 14, p. 5965, 2024.
- **Artigo Fundador:** HARRIS, F. W. _How Many Parts to Make at Once_. _Factory, The Magazine of Management_, v. 10, n. 2, p. 135–136, 1913.

**Recorte Aplicado:** Na indústria de placas de vídeo, componentes críticos possuem alto valor agregado e rápida depreciação devido ao lançamento contínuo de novas arquiteturas. O modelo adapta o Lote Econômico de Compra (EOQ) clássico incorporando uma taxa anual de obsolescência e custo de capital ($\theta$) sobre o valor unitário do kit ($P$), sujeito às restrições físicas de pedido mínimo do fornecedor ($Q_{\min}$) e capacidade máxima do cofre climatizado da fábrica ($Q_{\max}$).

---

## 2. Persona e Cenário de Uso

- **Persona:** **Carlos Mendes**, 38 anos, Gerente de Supply Chain da **Amazonas GPU Tech** (planta industrial de montagem de placas de vídeo).
- **Dor / Problema:** Comprar lotes pequenos com muita frequência eleva os gastos com frete aéreo internacional e desembaraço aduaneiro ($S$). Por outro lado, compras volumosas imobilizam capital de giro, lotam o cofre antiestático ($Q_{\max}$) e expõem o estoque ao risco de desvalorização tecnológica ($\theta$).
- **Métricas de Sucesso:** Redução do custo logístico total anual em relação à política empírica atual (compras mensais fixas) e garantia matemática de viabilidade física de armazenagem.

---

## 3. Modelagem Matemática

### 3.1. Variável de Decisão e Domínio

- **Variável de decisão:** $Q \in \mathbb{R}_{>0}$ (quantidade de kits GPU + VRAM adquiridos por pedido de importação).
- **Domínio restrito:** $Q \in [Q_{\min}, Q_{\max}]$, onde $Q_{\min}$ é o MOQ (_Minimum Order Quantity_) do fabricante e $Q_{\max}$ é o limite físico do cofre climatizado.

### 3.2. Função Objetivo — Custo Total Anual $C(Q)$

$$C(Q) = \underbrace{D \cdot P}_{\text{Aquisição}} + \underbrace{\frac{D}{Q} \cdot S}_{\text{Pedidos e Aduana}} + \underbrace{\frac{Q}{2} \cdot (h + \theta \cdot P)}_{\text{Armazenagem + Obsolescência}}$$

Onde:

- $D$: Demanda anual de placas de vídeo (unidades/ano);
- $P$: Custo unitário do kit de componentes (R$);
- $S$: Custo fixo por pedido de importação (R$);
- $h$: Custo físico anual de armazenagem climatizada e seguro por unidade (R$/ano);
- $\theta$: Taxa anual de obsolescência tecnológica e custo de oportunidade do capital ($\text{a.a.}$).

### 3.3. Critérios de Otimização (1ª e 2ª Derivadas)

1. **Ponto Crítico ($C'(Q) = 0$):**
   $$C'(Q) = -\frac{D \cdot S}{Q^2} + \frac{h + \theta \cdot P}{2} = 0 \implies Q_{\text{crítico}} = \sqrt{\frac{2 \cdot D \cdot S}{h + \theta \cdot P}}$$

2. **Teste da Segunda Derivada (Concavidade):**
   $$C''(Q) = \frac{2 \cdot D \cdot S}{Q^3}$$
   Como $D > 0$, $S > 0$ e $Q > 0$, tem-se $C''(Q) > 0$ em todo o domínio. Logo, $C(Q)$ é **estritamente convexa** e $Q_{\text{crítico}}$ é um **mínimo global**.

3. **Análise de Fronteira:**
   - Se $Q_{\min} \le Q_{\text{crítico}} \le Q_{\max}$, então $Q^* = Q_{\text{crítico}}$.
   - Caso contrário, o sistema avalia $C(Q_{\min})$ e $C(Q_{\max})$ e seleciona o extremo viável de menor custo.

---

## 4. Arquitetura e Estrutura do Projeto

```text
projeto-gpu-eoq/
├── app.py                  # Back-end REST API (FastAPI + SymPy + SQLite + Logs)
├── index.html              # Front-end SPA (React + Tailwind CSS + Chart.js + MathJax)
├── experimento_sympy.py    # Script isolado para reprodução simbólica no terminal
├── requirements.txt        # Dependências do ambiente Python
├── simulacoes_gpu.db       # Banco de dados SQLite (gerado automaticamente na execução)
├── gpu_eoq_system.log      # Arquivo de logs de auditoria e erros (gerado automaticamente)
└── README.md               # Documentação técnica do projeto
```
