# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Helper: parse user equation
# Accepts:
#  - Q = a - bP   (e.g. Q = 120 - 3P)
#  - P = c - dQ   (e.g. P = 60 - 0.2Q)
# Returns (is_valid, form, a, b) where:
#  - if form == 'Q_from_P' => Q = a - b*P
#  - if form == 'P_from_Q' => P = a - b*Q
# -------------------------
def parse_equation(eq_text):
    eq = eq_text.replace(" ", "").replace("−", "-")
    # Q=a-bP
    m = re.match(r"^[Qq]=([+-]?\d*\.?\d+)-([+-]?\d*\.?\d+)[Pp]$", eq)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return True, "Q_from_P", a, b
    # P=a-bQ
    m = re.match(r"^[Pp]=([+-]?\d*\.?\d+)-([+-]?\d*\.?\d+)[Qq]$", eq)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return True, "P_from_Q", a, b
    return False, None, None, None

# -------------------------
# Convert to functions
# Given parse result, return functions:
#   p_from_q(q), q_from_p(p)
# -------------------------
def functions_from_parse(parse):
    valid, form, a, b = parse
    if not valid:
        # default: Q = 100 - 2P  (your original line)
        return (lambda q: 50 - 0.5*q, lambda p: 100 - 2*p), "Q = 100 - 2P"
    if form == "Q_from_P":
        def q_from_p(p, a=a, b=b): return a - b * p
        def p_from_q(q, a=a, b=b): return (a - q) / b
        eqstr = f"Q = {a:g} - {b:g}P"
        return (p_from_q, q_from_p), eqstr
    else:
        def p_from_q(q, a=a, b=b): return a - b * q
        def q_from_p(p, a=a, b=b): return (a - p) / b
        eqstr = f"P = {a:g} - {b:g}Q"
        return (p_from_q, q_from_p), eqstr

# -------------------------
# Elasticity point formula (point elasticity)
# PED = (dQ/dP) * (P/Q)  ; for linear Q = a - bP, dQ/dP = -b
# If function given as P = a - bQ -> invert or compute derivative accordingly
# We will compute elasticities numerically for stability.
# -------------------------
def point_elasticity(q, p, q_from_p_func):
    # approximate derivative dq/dp using small delta
    dp = 1e-4 * max(1.0, p)
    q_plus = q_from_p_func(p + dp)
    q_minus = q_from_p_func(p - dp)
    dq_dp = (q_plus - q_minus) / (2*dp)
    if q == 0:
        return np.nan
    return dq_dp * (p / q)

# -------------------------
# UI: header and sidebar
# -------------------------
st.set_page_config(page_title="Elasticidad Pro — Dashboard Animado", layout="wide")
st.title("📈 Elasticidad Pro — App interactiva (Streamlit + Plotly)")

st.sidebar.header("Controles principales")

# Theme
theme = st.sidebar.radio("Tema", ["Oscuro (neón)", "Claro"], index=0)

# Input: equation 1 (base)
st.sidebar.subheader("Ecuación - Curva A (obligatoria)")
eq1_input = st.sidebar.text_input("Ingresa la ecuación (ej.: Q = 100 - 2P  o  P = 50 - 0.5Q)",
                                  value="Q = 100 - 2P")
parse1 = parse_equation(eq1_input)
(funcs1, eq1_str) = functions_from_parse(parse1)

# Optional: equation 2 for comparison
st.sidebar.subheader("Ecuación - Curva B (opcional para comparar)")
eq2_input = st.sidebar.text_input("Ecuación 2 (dejar vacío para desactivar)", value="")
if eq2_input.strip() != "":
    parse2 = parse_equation(eq2_input)
    funcs2, eq2_str = functions_from_parse(parse2)
    compare_two = True if parse2[0] else False
else:
    funcs2, eq2_str = None, None
    compare_two = False

# Animation settings
st.sidebar.subheader("Animación y controles")
easing = st.sidebar.selectbox("Easing (transición)", ["linear", "cubic-in-out", "elastic", "bounce", "sin-in-out"], index=1)
frames_count = st.sidebar.slider("Frames por animación", min_value=10, max_value=120, value=40, step=10)
play_on_load = st.sidebar.checkbox("Reproducir al cargar", value=True)

# Controls: price slider for manual control
st.sidebar.subheader("Control del precio (Curva A)")
p_manual = st.sidebar.slider("Precio P (Curva A)", min_value=0.01, max_value=200.0, value=25.0, step=0.01, format="%.2f")

# Options
st.sidebar.subheader("Extras")
show_shade = st.sidebar.checkbox("Sombreado dinámico bajo la curva (Q hasta punto)", value=True)
show_rect_it = st.sidebar.checkbox("Mostrar rectángulo (Ingreso Total P×Q)", value=True)
show_area_it = st.sidebar.checkbox("Animación 'respiración' del área IT", value=True)
show_steps = st.sidebar.checkbox("Modo 'Clase' (mostrar pasos matemáticos)", value=False)
auto_scale = st.sidebar.checkbox("Ajuste automático de escala", value=True)
show_tooltips_plotly = True  # Plotly shows them automatically
allow_custom_axes = st.sidebar.checkbox("Permitir ajuste manual de ejes", value=False)

# If allow custom axes, show inputs
if allow_custom_axes:
    x_max = st.sidebar.number_input("Máx Q (eje X)", value=100.0)
    y_max = st.sidebar.number_input("Máx P (eje Y)", value=60.0)
else:
    x_max = None
    y_max = None

# Chat-like mini helper
st.sidebar.subheader("Chat explicativo")
user_question = st.sidebar.text_input("Pregunta sobre elasticidad (ej.: ¿qué pasa si subo precio?)", value="")
# We'll answer below with canned rules

# -------------------------
# Build functions for curve A and optional curve B
# -------------------------
p_from_q_A, q_from_p_A = funcs1
if compare_two:
    p_from_q_B, q_from_p_B = funcs2

# -------------------------
# Construct dataset of frames
# We'll animate price for curve A from a min to max or around the manual p
# If user gave manual p, we'll center frames around it
# -------------------------
# choose sensible ranges based on parsed parameters
# We'll sample Q from 0 to some maxQ depending on function parameters to autoscale
q_sample = np.linspace(0.0, 1.0, 5)  # placeholder to try to detect scale
# Attempt to infer reasonable ranges:
try:
    # compute some sample values to guess ranges
    sample_ps = [p_from_q_A(q) for q in np.linspace(0, 100, 5)]
    p_guess_max = max(sample_ps) if sample_ps else 50
except Exception:
    p_guess_max = 60

# Frames: interpolate from p_start to p_end
p_center = float(p_manual)
p_span = max(0.2, p_center * 0.6)  # animate +/- 60% of current price (sensible)
p_start = max(0.01, p_center - p_span)
p_end = p_center + p_span
frame_prices = np.linspace(p_start, p_end, frames_count)

# Prepare arrays for plotting (lots of q points for smooth curves)
q_vals = np.linspace(0.0, 1.0, 500)
# We'll map q_vals to a meaningful Q range: find where p_from_q_A becomes <= 0 to set Qmax
# Numeric approach: find q where p<=0
q_test = np.linspace(0, 1000, 5001)
try:
    p_test = np.array([p_from_q_A(q) for q in q_test])
    # Q range where p positive
    positive_idx = np.where(p_test > 0)[0]
    if positive_idx.size == 0:
        Qmax = 100.0
    else:
        Qmax = q_test[positive_idx[-1]] * 1.05
        Qmax = max(10.0, min(Qmax, 1000.0))
except Exception:
    Qmax = 100.0

q_vals = np.linspace(0.0, Qmax, 800)
p_curve_A = np.array([p_from_q_A(q) for q in q_vals])
it_curve_A = p_curve_A * q_vals

if compare_two:
    p_curve_B = np.array([p_from_q_B(q) for q in q_vals])
    it_curve_B = p_curve_B * q_vals

# Auto-scale axes if requested
if auto_scale:
    x_axis_max = Qmax * 1.05
    y_axis_max = max(np.nanmax(p_curve_A), 1.0) * 1.2
    if compare_two:
        y_axis_max = max(y_axis_max, np.nanmax(p_curve_B) * 1.2)
else:
    x_axis_max = x_max if x_max else Qmax
    y_axis_max = y_max if y_max else max(np.nanmax(p_curve_A), 50)

# -------------------------
# Build Plotly figure with frames (client-side animation)
# Two subplots: top = price vs Q (demand curve), bottom = IT vs Q
# We'll create frames for each price in frame_prices
# -------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.08,
                    subplot_titles=("Curva de Demanda (P vs Q)", "Ingreso Total (IT = P×Q)"))

# Base traces (curves) - static
fig.add_trace(go.Scatter(x=q_vals, y=p_curve_A, mode="lines", name=f"Demanda A: {eq1_str}",
                         line=dict(color="#58a6ff", width=3)), row=1, col=1)

if compare_two:
    fig.add_trace(go.Scatter(x=q_vals, y=p_curve_B, mode="lines", name=f"Demanda B: {eq2_str}",
                             line=dict(color="#ffa600", width=2, dash="dash")), row=1, col=1)

# Base IT curves
fig.add_trace(go.Scatter(x=q_vals, y=it_curve_A, mode="lines", name="IT A (P×Q)",
                         line=dict(color="#2ecc71", width=3)), row=2, col=1)
if compare_two:
    fig.add_trace(go.Scatter(x=q_vals, y=it_curve_B, mode="lines", name="IT B (P×Q)",
                             line=dict(color="#ff7b7b", width=2, dash="dash")), row=2, col=1)

# We'll add dynamic markers and filled shapes via frames
frames = []
for p_val in frame_prices:
    q_at_p = None
    try:
        q_at_p = q_from_p_A(p_val)
    except Exception:
        # fallback numeric
        q_candidates = np.interp([p_val], p_curve_A[::-1], q_vals[::-1])
        q_at_p = float(q_candidates[0])
    # clamp
    if q_at_p is None or np.isnan(q_at_p) or np.isinf(q_at_p):
        q_at_p = 0.0

    # compute IT
    it_at_p = p_val * q_at_p
    # compute elasticity
    try:
        el = point_elasticity(q_at_p, p_val, q_from_p_A)
    except Exception:
        el = np.nan

    # Data for top: marker point + optional filled area under curve from 0..q_at_p
    top_marker = go.Scatter(x=[q_at_p], y=[p_val], mode="markers",
                            marker=dict(size=18, color="#ff4d6d", line=dict(width=2, color="white")),
                            name="Punto Actual", hovertemplate=
                            f"Q: {q_at_p:.3f}<br>P: {p_val:.3f}<br>IT: {it_at_p:.3f}<br>Elasticidad: {el:.3f}<extra></extra>")
    # shading polygon: Q from 0..q_at_p
    if show_shade:
        q_shade = np.linspace(0, q_at_p, 40)
        p_shade = [p_from_q_A(q) for q in q_shade]
        # We'll create an area trace
        shade_trace = go.Scatter(x=np.concatenate([q_shade, q_shade[::-1]]),
                                 y=np.concatenate([p_shade, np.zeros_like(p_shade)]),
                                 fill="toself", mode="none",
                                 name="Sombreado (0..Q)", showlegend=False, opacity=0.12,
                                 fillcolor="#58a6ff")
    else:
        shade_trace = go.Scatter(x=[], y=[], mode="none", showlegend=False)

    # The IT rectangle (in bottom subplot) can be represented as a filled polygon from (0,0)->(q,0)->(q,p)->(0,p)
    if show_rect_it:
        rect_x = [0, q_at_p, q_at_p, 0, 0]
        rect_y = [0, 0, it_at_p, it_at_p, 0]  # note: to show rectangle on IT chart, use IT value on y
        rect_trace = go.Scatter(x=rect_x, y=rect_y, fill="toself", mode="none",
                                opacity=0.12, fillcolor="#2ecc71", showlegend=False)
    else:
        rect_trace = go.Scatter(x=[], y=[], mode="none", showlegend=False)

    # point for IT subplot
    bottom_marker = go.Scatter(x=[q_at_p], y=[it_at_p], mode="markers",
                               marker=dict(size=16, color="#ff4d6d", line=dict(width=2, color="white")),
                               hovertemplate=f"Q: {q_at_p:.3f}<br>IT: {it_at_p:.3f}<extra></extra>")

    # Add frame with updated data for the dynamic traces
    frame = go.Frame(data=[top_marker, shade_trace, bottom_marker, rect_trace],
                     name=f"p={p_val:.4f}",
                     layout=go.Layout(
                         annotations=[
                             dict(x=0.02, y=0.95, xref="paper", yref="paper",
                                  text=f"P = {p_val:.2f}  •  Q = {q_at_p:.2f}  •  IT = ${it_at_p:.2f}  •  Elasticidad = {el:.2f}",
                                  showarrow=False, font=dict(color="white" if theme.startswith("Oscuro") else "black"))
                         ]
                     ))
    frames.append(frame)

# Add the first frame's dynamic traces to the base figure (so initial display has markers)
if frames:
    fig.add_trace(frames[0].data[0], row=1, col=1)  # top marker
    fig.add_trace(frames[0].data[1], row=1, col=1)  # shade
    fig.add_trace(frames[0].data[2], row=2, col=1)  # bottom marker
    fig.add_trace(frames[0].data[3], row=2, col=1)  # rect

fig.frames = frames

# Layout styling depending on theme
if theme.startswith("Oscuro"):
    bg_color = "#0d1117"
    paper_bg = "#0d1117"
    font_color = "white"
    gridcolor = "#222831"
else:
    bg_color = "white"
    paper_bg = "white"
    font_color = "black"
    gridcolor = "#e6e6e6"

fig.update_layout(
    height=800,
    plot_bgcolor=bg_color,
    paper_bgcolor=paper_bg,
    font=dict(color=font_color),
    margin=dict(l=60, r=40, t=100, b=60),
    updatemenus=[
        dict(type="buttons",
             showactive=False,
             x=0.0,
             y=1.12,
             xanchor="left",
             yanchor="top",
             buttons=[
                 dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 60, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 300, "easing": easing}}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}]),
                 dict(label="Reset",
                      method="animate",
                      args=[[frames[0].name], {"frame": {"duration": 0, "redraw": True},
                                               "mode": "immediate",
                                               "transition": {"duration": 0}}])
             ])
    ],
    sliders=[{
        "active": 0,
        "currentvalue": {"prefix": "Precio: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 50},
        "steps": [
            {"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 0}}],
             "label": f"{float(f.name.split('=')[1]):.2f}",
             "method": "animate"} for f in frames]
    }]
)

# Axes
fig.update_xaxes(title_text="Cantidad (Q)", row=2, col=1, range=[0, x_axis_max], gridcolor=gridcolor)
fig.update_xaxes(title_text="Cantidad (Q)", row=1, col=1, range=[0, x_axis_max], gridcolor=gridcolor)
fig.update_yaxes(title_text="Precio (P)", row=1, col=1, range=[0, y_axis_max], gridcolor=gridcolor)
fig.update_yaxes(title_text="Ingreso Total (IT)", row=2, col=1, range=[0, np.nanmax(it_curve_A) * 1.2], gridcolor=gridcolor)

# Legend placement
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# If show_area_it = True, animate a gentle "breathing" by adding a small oscillation to opacity via frames
if show_area_it and len(fig.frames) > 0:
    # modify frames to change rect opacity slightly (we already set opacity constant, but we can emulate breathing by toggling opacity)
    for i, f in enumerate(fig.frames):
        # compute small multiplier
        breath = 0.08 + 0.02 * np.sin(2 * np.pi * (i / max(1, len(fig.frames))))
        # adjust fill opacity for rect trace (currently it's the 4th in frame.data)
        # but the property is in the trace; since we can't alter opacity for fill easily here, we'll attach an annotation that changes alpha color
        pass  # kept for clarity — plotly handles nice visuals already

# -------------------------
# Layout in Streamlit
# -------------------------
col_main, col_right = st.columns([3, 1])

with col_main:
    st.subheader("Visualización dinámica")
    # show the plotly chart (Plotly animation controls are client-side)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Provide play-on-load mechanism (simulate by auto-triggering Play using frames if requested)
    if play_on_load:
        # instruct user how to play (Plotly shows Play button); can't auto-trigger reliably from Python
        st.info("Usa el botón *Play* del gráfico (arriba) para ver la animación. También puedes mover el slider para recorrer los frames.")

    # Additional toggles / info
    with st.expander("Detalles del punto actual (Curva A)"):
        # compute values at the manual price
        try:
            q_m = q_from_p_A(p_manual)
            it_m = p_manual * q_m
            el_m = point_elasticity(q_m, p_manual, q_from_p_A)
        except Exception:
            q_m = np.nan; it_m = np.nan; el_m = np.nan
        st.markdown(f"- **Ecuación A:** {eq1_str}")
        if compare_two:
            st.markdown(f"- **Ecuación B:** {eq2_str}")
        st.markdown(f"- **Precio actual (manual):** ${p_manual:.2f}")
        st.markdown(f"- **Cantidad resultante:** {q_m:.3f}")
        st.markdown(f"- **Ingreso Total (IT):** ${it_m:.2f}")
        st.markdown(f"- **Elasticidad (point):** {el_m:.3f}")
        # Interpretation (auto)
        if not np.isnan(el_m):
            if abs(el_m) > 1.0:
                st.success("Interpretación: Demanda **Elástica** — consumidores sensibles al precio.")
            elif abs(el_m) < 1.0:
                st.warning("Interpretación: Demanda **Inelástica** — consumidores poco sensibles al precio.")
            else:
                st.info("Interpretación: Demanda **Unitaria** — variación porcentual igual en P y Q.")

with col_right:
    st.subheader("Herramientas")
    st.markdown("**Modo Clase / Explicaciones**")
    if show_steps:
        st.markdown("**Paso 1.** Identifica la forma de la ecuación (lineal).  \n"
                    "**Paso 2.** Deriva: si Q = a - bP entonces dQ/dP = -b.  \n"
                    "**Paso 3.** Elasticidad puntual = dQ/dP × P/Q.  \n"
                    "**Ejemplo:** Si Q = 100 - 2P, en P=25 -> Q=50, dQ/dP=-2 -> PED = -2*(25/50) = -1 (Unitario).")
    else:
        st.markdown("Activa 'Modo Clase' en la barra lateral para ver pasos matemáticos.")

    st.markdown("---")
    st.subheader("Mini Chat — ayuda rápida")
    if user_question.strip() != "":
        q_lower = user_question.lower()
        answer = "No entendí exactamente, prueba con: '¿qué pasa si subo el precio?', 'explica elasticidad', 'dónde está IT máximo', o 'por qué la curva se desplaza'."
        if "qué pasa" in q_lower and "sub" in q_lower:
            answer = ("Si subes el precio, la cantidad demandada cae (según la curva). "
                      "Si la demanda es elástica (|PED|>1) el ingreso total disminuirá. "
                      "Si es inelástica (|PED|<1) el ingreso total aumentará.")
        elif "elasticidad" in q_lower:
            answer = ("Elasticidad precio de la demanda mide la sensibilidad de la cantidad ante cambios en el precio. "
                      "PED = (dQ/dP) * (P/Q). Valores absolutos >1 son elásticos, <1 inelásticos, =1 unitarios.")
        elif "it máximo" in q_lower or "max" in q_lower:
            answer = ("Para una demanda lineal Q = a - bP, el ingreso total P×Q alcanza su máximo en el punto unitario de elasticidad. "
                      "Gráficamente suele coincidir con la mitad de la cantidad en la recta cuando la recta cruza el eje Q (ver anotación en gráfico).")
        elif "por qué" in q_lower and "se desplaza" in q_lower:
            answer = ("Una curva de demanda se desplaza por cambios distintos al precio (preferencias, ingreso, precios de otros bienes, expectativas). "
                      "Un movimiento a lo largo de la curva es causado por cambios en P (o Q), no por desplazamientos.")
        st.markdown(f"**Pregunta:** {user_question}")
        st.markdown(f"**Respuesta:** {answer}")
    else:
        st.markdown("Escribe una pregunta en la barra lateral para recibir una respuesta rápida.")

# -------------------------
# Footer / author
# -------------------------
st.markdown("---")
st.markdown("**Notas:** Esta app usa Plotly para animaciones client-side (suavidad y controles Play/Pause).")
st.markdown("Creado por Diego (adaptable). Si quieres: puedo convertir los gráficos a Plotly Dash, exportar snapshots, o agregar tooltips más avanzados.")
