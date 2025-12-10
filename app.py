import re
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

KB_PATH = "knowledge_house_electricity.csv"
TARIFF_PATH = "eso_tariffs.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

ASSUMED_CAPACITY_KW = 5.0


@st.cache_resource
def load_models_and_kb():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    qa_pipeline = pipeline(
        "question-answering",
        model=QA_MODEL_NAME,
        tokenizer=QA_MODEL_NAME,
    )

    df = pd.read_csv(KB_PATH, encoding="utf-8", sep=";")
    df = df[["question", "answer"]].dropna()

    questions = df["question"].tolist()
    question_embeddings = embedder.encode(questions, convert_to_tensor=True)

    tariff_df = pd.read_csv(TARIFF_PATH, encoding="utf-8", sep=";")

    return embedder, qa_pipeline, df, question_embeddings, tariff_df


embedder, qa_pipeline, df_kb, question_embeddings, tariff_df = load_models_and_kb()


def retrieve_relevant_context(user_question: str, top_k: int = 1) -> str:
    if len(df_kb) == 0:
        return ""

    top_k = min(top_k, len(df_kb))

    query_embedding = embedder.encode(user_question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = scores.topk(k=top_k)

    context_parts = []
    for idx in top_results.indices:
        idx = int(idx)
        q = str(df_kb.iloc[idx]["question"])
        a = str(df_kb.iloc[idx]["answer"])
        context_parts.append(f"Question: {q}\nAnswer: {a}")

    return "\n\n".join(context_parts)


def answer_from_knowledge(user_question: str) -> str:
    context = retrieve_relevant_context(user_question, top_k=1)

    if not context.strip():
        return "Sorry, I couldn't find an answer in my knowledge base."

    result = qa_pipeline(
        question=user_question,
        context=context,
        max_answer_len=300,
    )

    answer = result.get("answer", "").strip()

    for stop_token in ["Question:", "Answer:"]:
        if stop_token in answer:
            answer = answer.split(stop_token)[0].strip()

    if not answer:
        return "Sorry, I couldn't extract an answer from the context."

    return answer


def evaluate_consumption(kwh: float) -> str:
    if kwh < 150:
        return (
            f"If monthly electricity consumption is about {kwh:.1f} kWh.\n\n"
            "Your electricity consumption is **LOW**.\n"
            "This indicates efficient energy use or a small household."
        )
    elif 150 <= kwh <= 350:
        return (
            f"If monthly electricity consumption is about {kwh:.1f} kWh.\n\n"
            "Your electricity consumption is **NORMAL**.\n"
            "This is typical for homes without electric heating."
        )
    elif 350 < kwh <= 600:
        return (
            f"If monthly electricity consumption is about {kwh:.1f} kWh.\n\n"
            "Your electricity consumption is **HIGH**.\n"
            "You may be using powerful appliances or running them for long periods."
        )
    else:
        return (
            f"If monthly electricity consumption is about {kwh:.1f} kWh.\n\n"
            "Your electricity consumption is **VERY HIGH**.\n"
            "This often happens with electric heating or very intensive usage."
        )


def calculate_best_tariff(kwh: float) -> str:

    df = tariff_df.copy()

    df = df[df["tariff_type"] == "single"]

    if df.empty:
        return "Tariff data is not available."

    costs = {}

    for _, row in df.iterrows():
        plan = row["plan"]
        energy_price = row["energy_eur_per_kwh"]
        fixed_fee = row["fixed_eur_per_month"]
        capacity_price = row["capacity_eur_per_kw_month"]

        total = (
            kwh * energy_price
            + fixed_fee
            + ASSUMED_CAPACITY_KW * capacity_price
        )

        if plan in costs:
            costs[plan] = min(costs[plan], total)
        else:
            costs[plan] = total

    best_plan = min(costs, key=costs.get)
    best_price = costs[best_plan]


    sorted_plans = sorted(costs.items(), key=lambda x: x[1])

    lines = []
    lines.append(
        f"For your monthly consumption of **{kwh:.1f} kWh** "
        f"(assuming allowed capacity of {ASSUMED_CAPACITY_KW:.1f} kW and single time zone tariffs):"
    )
    for plan, price in sorted_plans:
        lines.append(f"- **{plan.title()}** plan: ~ **{price:.2f} € / month** (without VAT)")

    lines.append(f"\nBest choice: **{best_plan.title()}** (~{best_price:.2f} € / month, without VAT).")

    return "\n".join(lines)


def extract_kwh(text: str):
    text = text.lower().replace(",", ".")
    pattern = r"(\d+(\.\d+)?)\s*(kwh|kilowatt-hours?|kilowatt hour[s]?)"
    match = re.search(pattern, text)
    if match:
        value = float(match.group(1))
        return value
    return None


def extract_kwh_loose(text: str):
    text = text.lower().replace(",", ".")
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        value = float(match.group(1))
        return value
    return None


def chatbot_reply(user_text: str) -> str:
    kwh = extract_kwh(user_text)
    if kwh is None:
        kwh = extract_kwh_loose(user_text)

    if kwh is not None and kwh > 0:
        consumption_text = evaluate_consumption(kwh)
        tariff_text = calculate_best_tariff(kwh)
        return consumption_text + "\n\n" + tariff_text

    return answer_from_knowledge(user_text)



st.set_page_config(page_title="Home Electricity Consumption Bot", page_icon="💡")

st.title("Home Electricity Consumption Bot")
st.markdown(
    """
This bot uses **NLP models** to answer questions about household electricity consumption  
and to **evaluate your monthly kWh usage** and **recommend the cheapest distribution tariff**.

- Ask things like: `What is a kilowatt-hour?`  
- Or type: `My monthly consumption is about 250 kWh`  
"""
)

if "history" not in st.session_state:
    st.session_state["history"] = []


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your question or your monthly usage:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    bot_answer = chatbot_reply(user_input.strip())

    st.session_state["history"] = [
        ("Question", user_input.strip()),
        ("Answer", bot_answer),
    ]

for speaker, text in st.session_state["history"]:
    if speaker == "Question":
        st.markdown(f"**Question:** {text}")
    else:
        st.markdown(f"**Answer:** {text}")

st.markdown("---")
st.caption(
    f"Knowledge base size: {len(df_kb)} Q&A pairs · "
    f"Tariffs loaded: {len(tariff_df)} rows · "
    f"Embedding model: `{EMBEDDING_MODEL_NAME}` · "
    f"QA model: `{QA_MODEL_NAME}`"
)


