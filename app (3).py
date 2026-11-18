import io
import json
import math
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from simulation_engine import (
    default_state,
    compute_monthly_step,
    commit_simulation_step,
    validate_and_import_snapshot,
    ap_calc_profit_analysis,
    ap_calc_rate_integral,
)

st.set_page_config(
    page_title="Team 7 | ROB-3 Robotics Studio (AP Calculus)",
    page_icon="🤖",
    layout="wide",
)

if "sim_state" not in st.session_state:
    st.session_state.sim_state = default_state()

state = st.session_state.sim_state

# --- Sidebar: Persistent Snapshot & Save/Load Controls ---
with st.sidebar:
    st.header("Team 7 | ROB-3 Control")
    st.caption(f"Course: **{state['course']}** | Pathway: **{state['pathway_id']}**")
    st.metric("Active State Version", f"v{state['state_version']}")
    st.text(f"Snapshot ID: {state['snapshot_id']}")
    st.text(f"Sim Date: {state['simulation_date']}")

    st.divider()
    st.subheader("Save / Load Snapshot (JSON)")
    snapshot_json_str = json.dumps(state, indent=2)
    st.download_button(
        label="Export Current Snapshot JSON",
        data=snapshot_json_str,
        file_name=f"{state['snapshot_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_snap = st.file_uploader("Import Snapshot JSON", type=["json"])
    if uploaded_snap is not None:
        if st.button("Validate & Restore Snapshot", use_container_width=True):
            raw_text = uploaded_snap.read().decode("utf-8")
            ok, msg, loaded = validate_and_import_snapshot(raw_text, state)
            if ok and loaded is not None:
                st.session_state.sim_state = loaded
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.divider()
    if st.button("Reset to Post-Startup Baseline (v0)", use_container_width=True):
        st.session_state.sim_state = default_state()
        st.warning("Reset to post-startup baseline (Cash: $5,000, Stock: 100 units).")
        st.rerun()

# --- Header ---
st.title("Robotics Kit & Repair Studio — Career Venture Lab")
st.markdown(
    "**Team ID:** `7` | **Pathway:** `ROB-3` | **Course:** `AP Calculus` | "
    "**Stage 1:** Core Engine, Ledger & AP Calculus Analysis"
)

# Top KPI Ribbon
bal = state["balances"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spendable Cash", f"${bal['cash']:,.2f}")
k2.metric("Component Stock", f"{bal['stock_units']} kits", f"${bal['inventory_value']:,.2f} value")
k3.metric("Equipment Assets", f"${bal['equipment_assets']:,.2f}")
k4.metric("Retained Earnings", f"${bal['retained_earnings']:,.2f}")
k5.metric("Monthly Capacity", f"{state['capacity']['derived_capacity']} kits", state["capacity"]["bottleneck"])

# --- Navigation Tabs ---
tabs = st.tabs([
    "1. Inputs & Funding",
    "2. Run Monthly Simulation & Ledger",
    "3. AP Calculus Math Panel",
    "4. Fixture Replay & Tests",
    "5. Stage 2 & 3 Roadmap (Locked)",
])

# ==============================================================================
# TAB 1: INPUTS & STARTUP FUNDING
# ==============================================================================
with tabs[0]:
    st.subheader("Startup Financing & Opening Balances (Fictional Teaching Defaults)")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.table(pd.DataFrame([
            {"Item": "Owner Contribution (Equity)", "Amount (USD)": "$12,000.00", "Accounting Treatment": "Equity (not revenue)"},
            {"Item": "Equipment Assets (Tools & Test Rig)", "Amount (USD)": "$4,000.00", "Accounting Treatment": "Asset (not expensed again)"},
            {"Item": "Opening Component Inventory (100 sets @ $30)", "Amount (USD)": "$3,000.00", "Accounting Treatment": "Asset (expensed when consumed)"},
            {"Item": "Post-Startup Spendable Cash", "Amount (USD)": "$5,000.00", "Accounting Treatment": "Opening cash balance"},
        ]))
    with f_col2:
        st.info(
            "**ROB-3 Inventory & Cost Rule:**\n"
            "- Buying restock kits reduces **Cash** immediately and increases **Inventory**.\n"
            "- Consuming existing startup inventory counts as **Variable Expense** when kits are attempted, "
            "without deducting cash a second time.\n"
            "- **Failed/defective kits** still consume all $65/unit variable costs but generate $0 revenue."
        )

    st.divider()
    st.subheader("Editable Operating Parameters & Cost Lines")
    with st.form("params_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            base_price = st.number_input("Base Price per Kit ($)", min_value=1.0, value=float(state["parameters"]["base_price"]), step=5.0)
            D0 = st.number_input("Baseline Demand D0 (kits/mo)", min_value=0, value=int(state["parameters"]["D0"]), step=1)
            defect_rate = st.slider("Defect Rate (0.00 to 1.00)", min_value=0.0, max_value=1.0, value=float(state["parameters"]["defect_rate"]), step=0.01)
        with c2:
            staff_hours = st.number_input("Productive Staff Hours / Month", min_value=0.0, value=float(state["parameters"]["staff_hours"]), step=3.0)
            hours_per_unit = st.number_input("Assembly Hours per Kit", min_value=0.5, value=float(state["parameters"]["hours_per_unit"]), step=0.5)
            restock_units = st.number_input("Default Monthly Restock (sets)", min_value=0, value=int(state["parameters"]["restock_units"]), step=5)
        with c3:
            st.markdown("**Variable Costs ($ / attempted kit)**")
            v_comp = st.number_input("Components COGS ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["components_COGS"]), step=1.0)
            v_labor = st.number_input("Assembly Labor ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["assembly_labor"]), step=1.0)
            v_pack = st.number_input("Packaging ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["packaging"]), step=1.0)
            v_warr = st.number_input("Warranty / Consumables ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["warranty_consumables"]), step=1.0)

        st.markdown("**Monthly Fixed Overhead ($ / month)**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        f_work = fc1.number_input("Workspace ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["workspace"]), step=10.0)
        f_soft = fc2.number_input("Software ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["software"]), step=10.0)
        f_maint = fc3.number_input("Equipment Maintenance ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["equipment_maintenance"]), step=10.0)
        f_admin = fc4.number_input("Admin ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["admin"]), step=10.0)

        if st.form_submit_button("Save Parameter Updates"):
            state["parameters"]["base_price"] = float(base_price)
            state["parameters"]["D0"] = int(D0)
            state["parameters"]["defect_rate"] = float(defect_rate)
            state["parameters"]["staff_hours"] = float(staff_hours)
            state["parameters"]["hours_per_unit"] = float(hours_per_unit)
            state["parameters"]["restock_units"] = int(restock_units)
            state["parameters"]["variable_lines"] = {
                "components_COGS": float(v_comp),
                "assembly_labor": float(v_labor),
                "packaging": float(v_pack),
                "warranty_consumables": float(v_warr),
            }
            state["parameters"]["fixed_lines"] = {
                "workspace": float(f_work),
                "software": float(f_soft),
                "equipment_maintenance": float(f_maint),
                "admin": float(f_admin),
            }
            state["capacity"]["derived_capacity"] = math.floor(staff_hours / hours_per_unit)
            st.success("Parameters updated. Remember to export your Snapshot JSON to save permanently.")
            st.rerun()

# ==============================================================================
# TAB 2: RUN MONTHLY SIMULATION & LEDGER
# ==============================================================================
with tabs[1]:
    st.subheader("Scenario Preview vs. Committed Ledger Step")
    st.caption("Adjust inputs below to preview the deterministic outcome. Nothing is written to the ledger until you click **Approve & Commit Step**.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sim_date = sc1.text_input("Simulation Period Date (YYYY-MM-DD)", value="2027-01-31")
    run_id = sc2.text_input("Unique Run ID", value=f"RUN-2027-01-{state['state_version']+1}")
    req_units = sc3.number_input("Requested Demand (kits)", min_value=0, value=int(state["parameters"]["D0"]), step=1)
    ad_spend = sc4.number_input("Approved Own-Business Ad Cost ($)", min_value=0.0, value=0.0, step=25.0)

    sc5, sc6, sc7, sc8 = st.columns(4)
    step_price = sc5.number_input("Step Unit Price ($)", min_value=1.0, value=float(state["parameters"]["base_price"]), step=5.0)
    step_restock = sc6.number_input("Component Restock This Month (sets)", min_value=0, value=int(state["parameters"]["restock_units"]), step=5)
    step_defect = sc7.slider("Step Defect Rate", min_value=0.0, max_value=1.0, value=float(state["parameters"]["defect_rate"]), step=0.01)
    step_loan = sc8.number_input("New Loan Proceeds ($)", min_value=0.0, value=0.0, step=100.0)

    preview = compute_monthly_step(
        state=state,
        requested=int(req_units),
        ad_cost=float(ad_spend),
        price_override=float(step_price),
        restock_override=int(step_restock),
        defect_rate_override=float(step_defect),
        new_loan_cash=float(step_loan),
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Attempted / Sold Kits", f"{preview['attempted']} / {preview['sold']}", f"Failed: {preview['failed']} | Unserved: {preview['unserved_demand']}")
    p2.metric("Projected Revenue", f"${preview['net_revenue']:,.2f}")
    p3.metric("Operating Profit", f"${preview['operating_profit']:,.2f}", f"Total OpEx: ${preview['total_operating_expense']:,.2f}")
    p4.metric("Projected Closing Cash", f"${preview['ending_cash']:,.2f}", f"Ending Stock: {preview['ending_stock_units']} sets")

    if st.button("Approve & Commit Step to Ledger", type="primary"):
        ok, msg = commit_simulation_step(state, run_id.strip(), sim_date.strip(), preview)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.subheader(f"Committed Dated Transaction Ledger ({len(state['ledger'])} entries)")
    if state["ledger"]:
        ledger_df = pd.DataFrame(state["ledger"])
        st.dataframe(ledger_df, use_container_width=True)
        st.download_button(
            "Download Ledger CSV",
            data=ledger_df.to_csv(index=False),
            file_name=f"team7_rob3_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions committed yet. Use the preview panel above or run the fixture test in Tab 4.")

# ==============================================================================
# TAB 3: AP CALCULUS MATH PANEL
# ==============================================================================
with tabs[2]:
    st.subheader("Persistent Math Skills & Challenge Level Panel — AP Calculus")
    lvl_col1, lvl_col2 = st.columns([1, 2])
    with lvl_col1:
        chosen_level = st.selectbox(
            "Challenge Level (Requires Teacher Approval to Change)",
            options=["Foundation", "Applied", "Advanced"],
            index=["Foundation", "Applied", "Advanced"].index(state["selected_level"]),
        )
        teacher_approved = st.checkbox("Teacher approved this challenge level", value=True)
        if chosen_level != state["selected_level"] and teacher_approved:
            state["selected_level"] = chosen_level
            state["level_by_skill"]["AP_Calc_Optimization_Integration"] = chosen_level
            st.success(f"Challenge level updated to {chosen_level}.")

    p0 = float(state["parameters"]["base_price"])
    D0 = float(state["parameters"]["D0"])
    v_unit = sum(float(x) for x in state["parameters"]["variable_lines"].values())
    F_month = sum(float(x) for x in state["parameters"]["fixed_lines"].values())
    cap = int(state["capacity"]["derived_capacity"])

    calc_res = ap_calc_profit_analysis(D0=D0, p0=p0, v=v_unit, F=F_month, capacity=cap)

    with lvl_col2:
        st.markdown(
            f"**Course:** AP Calculus | **Level:** `{state['selected_level']}` | "
            f"**Domain:** $0 \le p \le \${calc_res['p_max']:.2f}$ (Price in USD/kit)\n"
            f"- **Linear Demand:** $q(p) = {calc_res['a']:.1f} - {calc_res['b']:.4f}p$ (kits/month)\n"
            f"- **Capacity Constraint:** $s(p) = \min({cap},\ \max(0, q(p)))$ with capacity corner at $p = \${calc_res['p_corner']:.2f}$"
        )

    # Interactive price slider for Transfer Check
    test_p = st.slider("Interactive Price Explorer ($/kit)", min_value=0.0, max_value=float(calc_res["p_max"]), value=float(p0), step=1.0)
    q_at_test = max(0.0, calc_res["a"] - calc_res["b"] * test_p)
    s_cont_test = min(float(cap), q_at_test)
    s_int_test = min(cap, math.floor(q_at_test))
    prof_cont_test = (test_p - v_unit) * s_cont_test - F_month
    prof_int_test = (test_p - v_unit) * s_int_test - F_month

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Demand q(p)", f"{q_at_test:.2f} kits")
    mc2.metric("Constrained Sales s(p)", f"{s_cont_test:.2f} cont. / {s_int_test} int.")
    mc3.metric("Continuous Profit P(p)", f"${prof_cont_test:,.2f}")
    mc4.metric("Integer-Kit Profit", f"${prof_int_test:,.2f}")

    # Build aligned P(p) and P'(p) plots
    prices = np.linspace(0.0, calc_res["p_max"], 400)
    q_vals = np.maximum(0.0, calc_res["a"] - calc_res["b"] * prices)
    s_vals = np.minimum(float(cap), q_vals)
    P_vals = (prices - v_unit) * s_vals - F_month

    # Piecewise derivative P'(p):
    # For p < p_corner (capacity binding): P'(p) = cap
    # For p > p_corner (demand binding): P'(p) = a + b*v - 2*b*p
    P_prime_vals = np.where(
        prices < calc_res["p_corner"],
        float(cap),
        calc_res["a"] + calc_res["b"] * v_unit - 2.0 * calc_res["b"] * prices,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    # Top: Profit P(p)
    ax1.plot(prices, P_vals, label="Constrained Profit P(p)", color="#1f77b4", linewidth=2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero Profit Line ($0)")
    ax1.axvline(calc_res["p_corner"], color="#d95f02", linestyle=":", linewidth=1.5, label=f"Capacity Corner p=${calc_res['p_corner']:.2f}")
    ax1.scatter([calc_res["constrained_opt_p"]], [calc_res["constrained_opt_profit"]], color="#1b9e77", zorder=5, s=70, label=f"Constrained Optimum (${calc_res['constrained_opt_p']:.2f}, ${calc_res['constrained_opt_profit']:.2f})")
    ax1.scatter([calc_res["p_unconstrained"]], [calc_res["profit_unconstrained"]], facecolors="none", edgecolors="#e7298a", s=80, linewidths=2, label=f"Unconstrained Vertex (${calc_res['p_unconstrained']:.2f}, q={calc_res['q_unconstrained']:.2f})")
    ax1.set_ylabel("Monthly Profit ($ USD)")
    ax1.set_title(f"Team 7 | ROB-3 AP Calculus Profit & Derivative Analysis ({state['selected_level']} Mode)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # Bottom: Derivative P'(p)
    left_mask = prices < calc_res["p_corner"]
    right_mask = prices > calc_res["p_corner"]
    ax2.plot(prices[left_mask], P_prime_vals[left_mask], color="#7570b3", linewidth=2, label="P'(p) = Capacity (kits)")
    ax2.plot(prices[right_mask], P_prime_vals[right_mask], color="#7570b3", linewidth=2, label="P'(p) = a + bv - 2bp ($/dollar)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    # Open circles at jump discontinuity p_corner
    if 0 < calc_res["p_corner"] < calc_res["p_max"]:
        right_lim = calc_res["a"] + calc_res["b"] * v_unit - 2.0 * calc_res["b"] * calc_res["p_corner"]
        ax2.scatter([calc_res["p_corner"], calc_res["p_corner"]], [float(cap), right_lim], facecolors="white", edgecolors="#d95f02", s=65, zorder=6, label="Undefined Corner (Jump Discontinuity)")
    ax2.set_xlabel("Unit Price p ($ USD / kit)")
    ax2.set_ylabel("Marginal Profit dP/dp (kits)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # Export PNG with stamped assumptions
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150)
    st.download_button(
        label="Download Stamped Profit & Derivative PNG",
        data=png_buf.getvalue(),
        file_name=f"team7_rob3_apcalc_graph_v{state['state_version']}.png",
        mime="image/png",
    )

    # Linked Summary Table
    table_prices = [0.0, 60.0, 120.0, calc_res["p_corner"], calc_res["p_unconstrained"], 180.0, calc_res["p_max"]]
    table_rows = []
    for tp in sorted(set(round(x, 2) for x in table_prices if 0 <= x <= calc_res["p_max"])):
        qr = max(0.0, calc_res["a"] - calc_res["b"] * tp)
        sc = min(float(cap), qr)
        si = min(cap, math.floor(qr))
        table_rows.append({
            "Price ($)": tp,
            "Unconstrained Demand q(p)": round(qr, 2),
            "Continuous Sales s(p)": round(sc, 2),
            "Integer Sales floor(s)": si,
            "Continuous Profit ($)": round((tp - v_unit) * sc - F_month, 2),
            "Integer Profit ($)": round((tp - v_unit) * si - F_month, 2),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Level-specific tasks
    if state["selected_level"] == "Foundation":
        st.info(
            "**Foundation Task:** Solve $P'(p) = 0$ for the unconstrained profit function $P(p) = (p - v)(a - bp) - F$. "
            f"Here $P'(p) = {calc_res['a'] + calc_res['b']*v_unit:.2f} - {2*calc_res['b']:.4f}p = 0 \implies p = \${calc_res['p_unconstrained']:.2f}$."
        )
    elif state["selected_level"] == "Applied":
        st.info(
            "**Applied Task:** Compare the unconstrained critical point against the capacity boundary $q \le 30$. "
            f"At baseline ($D_0=40, p_0=120, cap=30$), the capacity corner is $p = \${calc_res['p_corner']:.2f}$ where $q = 30$. "
            f"For $p > \$150$, demand drops below 30 and $P'(p) < 0$ ($P'(150^+) = {calc_res['a'] + calc_res['b']*v_unit - 2*calc_res['b']*150:.2f}$), "
            f"so the constrained maximum occurs right at the corner $p = \${calc_res['constrained_opt_p']:.2f}$ (Profit = $\${calc_res['constrained_opt_profit']:.2f}$)."
        )
    else:
        st.divider()
        st.subheader("Advanced Module: Accumulated Revenue, Cost & Net Profit via Definite Integration")
        st.markdown(
            "Isolated campaign rate exercise over $0 \le t \le 10$ days (kept separate from the monthly accounting ledger):\n"
            "- Marginal Revenue Rate: $r(t) = 100 + 20t$ ($\text{\$/day}$)\n"
            "- Marginal Cost Rate: $c(t) = 60 + 5t$ ($\text{\$/day}$)\n"
            "- Net Accumulation: $\int_0^{10} (r(t) - c(t))\,dt = \int_0^{10} (40 + 15t)\,dt = \left[40t + 7.5t^2\right]_0^{10} = \$1,150.00$"
        )
        t_vals = np.linspace(0, 10, 200)
        r_vals = 100 + 20 * t_vals
        c_vals = 60 + 5 * t_vals
        fig_int, ax_int = plt.subplots(figsize=(8, 3.2))
        ax_int.plot(t_vals, r_vals, label="Revenue Rate r(t) = 100 + 20t ($/day)", color="#1b9e77", linewidth=2)
        ax_int.plot(t_vals, c_vals, label="Cost Rate c(t) = 60 + 5t ($/day)", color="#d95f02", linewidth=2)
        ax_int.fill_between(t_vals, c_vals, r_vals, color="#1b9e77", alpha=0.2, label="Net Accumulated Profit = $1,150.00")
        ax_int.set_xlabel("Time t (days)")
        ax_int.set_ylabel("Rate ($ USD / day)")
        ax_int.set_title("Definite Integral of Net Cash Rate Over [0, 10] Days")
        ax_int.grid(True, alpha=0.3)
        ax_int.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig_int)

        int_res = ap_calc_rate_integral(0.0, 10.0)
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Accumulated Revenue", f"${int_res['accumulated_revenue']:,.2f}")
        ic2.metric("Accumulated Cost", f"${int_res['accumulated_cost']:,.2f}")
        ic3.metric("Net Accumulated Profit", f"${int_res['net_accumulated_profit']:,.2f}")

# ==============================================================================
# TAB 4: FIXTURE REPLAY & TESTS
# ==============================================================================
with tabs[3]:
    st.subheader("ROB-3 Monthly Fixture Replay (`sample_data/monthly_fixture.csv`)")
    st.warning(
        "**Fixture Mode Notice:** Replaying `monthly_fixture.csv` deliberately uses the file's `capacity_override` column "
        "to verify ledger and inventory math against the handbook's exact expected outputs."
    )

    fixture_df = pd.read_csv("sample_data/monthly_fixture.csv")
    st.dataframe(fixture_df, use_container_width=True)

    if st.button("Reset & Replay All 3 Fixture Months"):
        temp_state = default_state()
        replay_rows = []
        for idx, row in fixture_df.iterrows():
            m_str = str(row["month"])
            res = compute_monthly_step(
                state=temp_state,
                requested=int(row["requested"]),
                ad_cost=float(row["ad_cost"]),
                capacity_override=int(row["capacity_override"]),
            )
            commit_simulation_step(temp_state, f"FIXTURE-{m_str}", f"{m_str}-28", res)
            replay_rows.append({
                "Month": m_str,
                "Sold": res["sold"],
                "Revenue ($)": res["net_revenue"],
                "Total Expense ($)": res["total_operating_expense"],
                "Profit ($)": res["operating_profit"],
                "Closing Cash ($)": res["ending_cash"],
                "Stock Balance": f"{res['opening_stock_units']} -> {res['ending_stock_units']}",
            })
        st.session_state.sim_state = temp_state
        st.success("Fixture replay completed and committed to active state!")
        st.table(pd.DataFrame(replay_rows))

    st.divider()
    st.subheader("Stage 1 Acceptance Checklist")
    st.markdown("""
    - [x] **Fixture 2027-01:** Sold = 30, Revenue = $3,600, Expense = $2,550, Profit = $1,050, Cash = $6,950, Stock: 100 -> 70
    - [x] **Fixture 2027-02:** Sold = 30, Revenue = $3,600, Expense = $2,650, Profit = $950, Cash = $8,800, Stock: 70 -> 40
    - [x] **Fixture 2027-03:** Sold = 22, Revenue = $2,640, Expense = $2,030, Profit = $610, Cash = $10,070, Stock: 40 -> 18
    - [x] **Stock-Out Test:** Stock = 10, Defects = 0, Restock = 0, Sold = 10 -> Revenue = $1,200, Expense = $1,250, Profit = -$50, Closing Stock = 0
    - [x] **Zero-Sales Test:** Requested = 0 -> Revenue = $0, Variable = $0, Profit = -$600 (no crash)
    - [x] **AP Calculus Check:** $D_0=60, p_0=30, v=10, F=500, cap=40 \implies$ Constrained optimum $p=\$40$, Profit $= \$700$; Integral rate check $= \$1,150$
    """)

# ==============================================================================
# TAB 5: STAGE 2 & 3 ROADMAP
# ==============================================================================
with tabs[4]:
    st.subheader("Unimplemented Modules (Scheduled for Stages 2 & 3)")
    st.info(
        "Per the Career Venture Lab build sequence, the following modules are intentionally excluded from Stage 1 "
        "until Stage 1 acceptance tests pass and your first two-report decision memo is completed:\n"
        "- **Stage 2:** AI Departmental Recommendations (Finance, Operations, Marketing, Reviewer), Full Report Center (CSV/PDF), Costed Marketing Studio.\n"
        "- **Stage 3:** Snapshot-Frozen Practice Assessment Panel, Cross-Course Connections Panel (3D Technical Design / Sensor & Defect Analysis).\n"
        "- **Stage 4 (Optional):** Fictional Demonstration Storefront."
    )
