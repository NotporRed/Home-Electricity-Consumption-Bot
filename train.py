import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# NLP MODELIAI

# Semantinė paieška (sentence embeddings)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Question Answering modelis (DistilBERT SQuAD)
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL_NAME,
    tokenizer=QA_MODEL_NAME,
)

KB_PATH = "knowledge_house_electricity.csv"
df = pd.read_csv(KB_PATH, encoding="utf-8", sep=";")
df = df[["question", "answer"]].dropna()

questions = df["question"].tolist()

question_embeddings = embedder.encode(questions, convert_to_tensor=True)


def retrieve_relevant_context(user_question: str, top_k: int = 3) -> str:
    if len(df) == 0:
        return ""
    
    top_k = min(top_k, len(df))

    query_embedding = embedder.encode(user_question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = scores.topk(k=top_k)

    context_parts = []
    for idx in top_results.indices:
        idx = int(idx)
        q = str(df.iloc[idx]["question"])
        a = str(df.iloc[idx]["answer"])
        context_parts.append(f"Question: {q}\nAnswer: {a}")

    context = "\n\n".join(context_parts)
    return context


def answer_from_knowledge(user_question: str) -> str:
    context = retrieve_relevant_context(user_question, top_k=1)

    if not context.strip():
        return "Sorry, I couldn't find an answer in my knowledge base."

    result = qa_pipeline(
        question=user_question,
        context=context,
        max_answer_len=150,
    )

    return result.get("answer", "Sorry, I couldn't extract an answer.")

def evaluate_consumption(kwh: float) -> str:
    if kwh < 150:
        return (
            f"Your monthly electricity consumption is about {kwh:.1f} kWh.\n"
            "Your electricity consumption is LOW.\n"
            "This indicates efficient energy use or a small household."
        )
    elif 150 <= kwh <= 350:
        return (
            f"Your monthly electricity consumption is about {kwh:.1f} kWh.\n"
            "Your electricity consumption is NORMAL.\n"
            "This is typical for homes without electric heating."
        )
    elif 350 < kwh <= 600:
        return (
            f"Your monthly electricity consumption is about {kwh:.1f} kWh.\n"
            "Your electricity consumption is HIGH.\n"
            "You may be using powerful appliances or running them for long periods."
        )
    else:
        return (
            f"Your monthly electricity consumption is about {kwh:.1f} kWh.\n"
            "Your electricity consumption is VERY HIGH.\n"
            "This often happens with electric heating or very intensive usage."
        )


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

    if kwh is not None:
        return evaluate_consumption(kwh)

    return answer_from_knowledge(user_text)

if __name__ == "__main__":
    print(
        "Home electricity consumption bot.\n"
        "You can:\n"
        "   Ask questions about household electricity usage or enter your monthly usage (in English), e.g.:\n"
        "       'What is a kilowatt-hour?'\n"
        "       'How can I reduce my electricity consumption?'\n"
        "       'My consumption is about 250 kWh per month.'\n"
        "Type 'exit' or 'quit' to stop.\n"
    )

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower().strip() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chatbot_reply(user_input)
        print("Bot:", response)
