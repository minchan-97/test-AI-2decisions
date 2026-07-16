"""
streamlit_app.py — 토론 XAI: 두 AI의 '결심 순간'을 층별로 열어보기
====================================================================
아이디어:
  AI 둘이 어떤 사안을 두고 각자 입장을 낸다.
  그런데 "왜 그렇게 생각해?"라고 물으면 LLM은 사후 설명을 지어낸다(post-hoc).
  → 대신 **결론 토큰 직전의 내부**를 logit lens 로 열면,
    실제 계산 과정이 보인다. 이건 말과 달리 거짓말을 못 한다.

설계상 타협 (정직하게):
  - 작은 모델(distilgpt2/gpt2)은 제대로 된 논증을 못 한다. 논증은 유치하다.
  - 하지만 내부는 **진짜**다. API 모델은 논증이 좋아도 내부를 못 연다.
  - 그래서: 논증 품질을 포기하고 관찰 가능성을 택했다.

핵심 기법:
  자유 생성하면 순전파가 수백 번이라 볼 수 없다.
  → 결론을 **한 토큰**으로 고정(" yes"/" no")하고, 그 한 지점만 층별로 연다.
"""
import streamlit as st

st.set_page_config(page_title="토론 XAI — 결심의 층", layout="wide")

try:
    import torch
    import numpy as np
    import plotly.graph_objects as go
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    st.error(f"의존성 로드 실패: {e}")
    st.stop()

torch.set_num_threads(1)


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, output_hidden_states=True,
        low_cpu_mem_usage=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def _final_ln(model):
    base = getattr(model, "transformer", None) or getattr(model, "model", None)
    ln = getattr(base, "ln_f", None) if base is not None else None
    return ln if ln is not None else torch.nn.Identity()


@torch.no_grad()
def decision_lens(model, tok, prompt, options, apply_ln=True):
    """결론 토큰 직전에서, 선택지들의 확률을 층별로 추적.

    options: [" yes", " no"] 같은 후보 토큰 문자열
    returns: {option: [층별 확률]}, 층별 top1
    """
    ids = tok.encode(prompt, return_tensors="pt")
    out = model(ids)
    W_U = model.lm_head.weight
    ln_f = _final_ln(model)

    opt_ids = {}
    for o in options:
        enc = tok.encode(o)
        if enc:
            opt_ids[o] = enc[0]

    traj = {o: [] for o in opt_ids}
    top1 = []
    for h in out.hidden_states:
        v = h[0, -1, :]
        if apply_ln:
            v = ln_f(v)
        probs = torch.softmax(W_U @ v, dim=-1)
        for o, i in opt_ids.items():
            traj[o].append(float(probs[i]))
        ti = int(torch.argmax(probs))
        top1.append((tok.decode([ti]), float(probs[ti])))
    return traj, top1


def decided_layer(traj, options, margin=0.02):
    """어느 층에서 승자가 확정됐나 (마지막까지 안 뒤집힌 지점).

    margin: 이 차이 미만이면 '접전'으로 보고 굳었다고 판정하지 않는다.
            (확률이 거의 같은데 미세하게 앞섰다고 '초반부터 확신'이라
             표시하면 오해를 준다)
    """
    a, b = options[0], options[1]
    n = len(traj[a])
    winner = a if traj[a][-1] >= traj[b][-1] else b
    loser = b if winner == a else a
    layer = n - 1
    for i in range(n - 1, -1, -1):
        if traj[winner][i] - traj[loser][i] >= margin:
            layer = i
        else:
            break
    final_margin = traj[winner][-1] - traj[loser][-1]
    close = final_margin < margin
    return winner, layer, close


# ─────────────────────────────────────────────
st.title("토론 XAI — 두 AI의 '결심 순간'을 층별로 열기")
st.caption("말로 하는 설명은 지어낼 수 있다. 내부 계산은 못 지어낸다.")

with st.sidebar:
    st.markdown("### 토론자 설정")
    model_a = st.selectbox("AI-A 모델", ["distilgpt2", "gpt2"], index=0)
    model_b = st.selectbox("AI-B 모델", ["gpt2", "distilgpt2"], index=0)
    apply_ln = st.checkbox("최종 LayerNorm 적용", value=True,
                           help="끄면 초기 층이 깨지고 마지막 층이 정상화된다")
    st.markdown("---")
    st.caption("작은 모델이라 논증은 유치하다. "
               "대신 내부는 진짜로 열린다 — 이게 타협점.")

topic = st.text_input(
    "토론 사안 (영어, 예/아니오로 답할 수 있는 명제)",
    "Is it better to live in a big city than in a small town?")

c1, c2 = st.columns(2)
with c1:
    stance_a = st.text_input("AI-A 입장 유도", "You strongly support city life.")
with c2:
    stance_b = st.text_input("AI-B 입장 유도", "You strongly prefer small towns.")

options = [" yes", " no"]

if st.button("토론 시작 — 결심의 층 열기", type="primary"):
    with st.spinner("두 모델 로딩 + 순전파..."):
        tok_a, m_a = load_model(model_a)
        tok_b, m_b = load_model(model_b)

        # 결론을 한 토큰으로 고정하는 프롬프트
        p_a = f"{stance_a}\nQuestion: {topic}\nAnswer:"
        p_b = f"{stance_b}\nQuestion: {topic}\nAnswer:"

        traj_a, top_a = decision_lens(m_a, tok_a, p_a, options, apply_ln)
        traj_b, top_b = decision_lens(m_b, tok_b, p_b, options, apply_ln)

    # ── 결과 요약 ──
    wa, la, close_a = decided_layer(traj_a, options)
    wb, lb, close_b = decided_layer(traj_b, options)

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f"### AI-A ({model_a})")
        st.metric("결론", wa.strip(), f"{traj_a[wa][-1]:.1%}")
        if close_a:
            st.warning(f"접전 — 두 선택지 차이가 거의 없다. "
                       f"'결론'이라 부르기 어려운 상태.")
        else:
            st.write(f"**layer {la}** 부터 이 답으로 굳음 (총 {len(traj_a)-1}층 중)")
    with k2:
        st.markdown(f"### AI-B ({model_b})")
        st.metric("결론", wb.strip(), f"{traj_b[wb][-1]:.1%}")
        if close_b:
            st.warning(f"접전 — 두 선택지 차이가 거의 없다. "
                       f"'결론'이라 부르기 어려운 상태.")
        else:
            st.write(f"**layer {lb}** 부터 이 답으로 굳음 (총 {len(traj_b)-1}층 중)")

    if close_a or close_b:
        st.info("한쪽 이상이 접전이다. 작은 모델은 이런 판단을 잘 못한다 — "
                "그것 자체가 관찰 결과. 사안을 더 단순하게 바꿔보라.")
    elif wa == wb:
        st.info(f"두 AI가 같은 결론({wa.strip()})에 도달했다. "
                f"하지만 굳은 시점이 다르다 — A는 layer {la}, B는 layer {lb}. "
                f"**같은 답이어도 확신의 깊이가 다르다.**")
    else:
        st.success(f"두 AI가 다른 결론에 도달했다 — A는 {wa.strip()}, B는 {wb.strip()}. "
                   f"어느 층에서 갈라지기 시작했는지 아래 그래프에서 확인.")

    # ── 궤적 그래프 ──
    st.markdown("---")
    st.markdown("### 층별 확률 궤적 — 언제 마음을 정했나")
    g1, g2 = st.columns(2)
    for col, (traj, name, mdl) in [(g1, (traj_a, "AI-A", model_a)),
                                    (g2, (traj_b, "AI-B", model_b))]:
        with col:
            fig = go.Figure()
            for o, color in zip(options, ["seagreen", "indianred"]):
                fig.add_trace(go.Scatter(
                    x=list(range(len(traj[o]))), y=traj[o],
                    mode="lines+markers", name=o.strip(),
                    line=dict(color=color, width=3)))
            fig.update_layout(title=f"{name} ({mdl})",
                              xaxis_title="층 (0=embedding)",
                              yaxis_title="확률", height=380)
            st.plotly_chart(fig, use_container_width=True)

    # ── 층별 top1 (실제로 무슨 토큰을 생각했나) ──
    st.markdown("### 층별 최상위 예측 — 결론 말고 뭘 생각했나")
    d1, d2 = st.columns(2)
    for col, (top, name) in [(d1, (top_a, "AI-A")), (d2, (top_b, "AI-B"))]:
        with col:
            rows = [{"층": "embed" if i == 0 else f"layer {i}",
                     "최상위": f"{t.strip()!r}", "확률": f"{p:.1%}"}
                    for i, (t, p) in enumerate(top)]
            st.markdown(f"**{name}**")
            st.dataframe(rows, use_container_width=True, height=380)

    st.markdown("---")
    st.markdown("""
#### 이 화면이 XAI 인 이유

LLM 에게 "왜 그렇게 생각해?" 라고 물으면 **그럴듯한 설명을 지어낸다**(post-hoc rationalization).
그 설명이 실제 계산과 같다는 보장이 없다.

여기서 보는 건 다르다 — **실제 순전파의 중간 상태**다. 지어낼 수 없다.

**단, 정직하게:**
- 이건 '기여도 궤적'이지 **'분기'가 아니다.** 어느 층에서도 `if A then yes` 같은
  이산 결정은 일어나지 않는다. 두 선택지가 계속 확률을 나눠 갖다가 한쪽이 앞설 뿐.
- 마지막 층의 붕괴는 raw logit lens 의 한계다 (최종 LN 이 이미 적용된 상태에
  또 적용해서). LayerNorm 체크박스를 꺼서 비교해보라.
- 작은 모델이라 논증 자체는 유치하다. 관찰 가능성과 논증 품질을 맞바꾼 결과.
    """)
