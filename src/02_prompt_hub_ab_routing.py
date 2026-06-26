"""
Bước 2 — Prompt Hub + A/B Routing
==================================
  1. Tạo 2 prompt versions khác nhau về semantic:
     - V1: concise factual answer
     - V2: structured engineering answer
  2. Push cả 2 prompt lên LangSmith Prompt Hub
  3. Pull prompt từ Hub để dùng trong RAG chain
  4. Route request deterministically bằng MD5 hash của request_id
  5. Ghi log A/B routing vào evidence/02_ab_routing_log.txt
"""
import sys
import hashlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ Import config trước LangChain/LangSmith để env tracing được set sớm.
import config

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
    build_section_anchor_retriever,
)
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Prompt names trên LangSmith Prompt Hub ──────────────────────────────
PROMPT_V1_NAME = "day22-sdlc-vibe-coding-v1-concise"
PROMPT_V2_NAME = "day22-sdlc-vibe-coding-v2-structured"


# ── 2. Semantic prompt versions ────────────────────────────────────────────
SYSTEM_V1 = """
You are a concise factual RAG assistant answering questions about the paper
"The New SDLC With Vibe Coding".

Use only the provided context.
If the context does not contain the answer, say you do not know based on the provided context.

Answer style:
- Be direct and concise.
- Prefer one short paragraph or a short bullet list.
- Do not add extra implications unless the question asks for them.

Context:
{context}
""".strip()


SYSTEM_V2 = """
You are a structured engineering RAG assistant answering questions about the paper
"The New SDLC With Vibe Coding".

Use only the provided context.
If the context does not contain the answer, say you do not know based on the provided context.

Answer style:
- Structure the answer for a software engineering audience.
- When useful, organize the answer as:
  1. Definition
  2. Why it matters
  3. SDLC or engineering implication
- Keep the answer grounded in the retrieved context.
- Do not invent facts outside the context.

Context:
{context}
""".strip()


def create_prompt(system_message: str) -> ChatPromptTemplate:
    """
    Tạo ChatPromptTemplate có 2 biến:
      - context
      - question
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{question}"),
        ]
    )


def push_prompts_to_hub():
    """
    Push 2 prompt versions lên LangSmith Prompt Hub.
    Nếu prompt đã tồn tại, LangSmith sẽ tạo commit/version mới.
    """
    prompt_v1 = create_prompt(SYSTEM_V1)
    prompt_v2 = create_prompt(SYSTEM_V2)

    print("📤 Pushing prompts to LangSmith Prompt Hub ...")

    url_v1 = hub.push(
        PROMPT_V1_NAME,
        prompt_v1,
        new_repo_is_public=False,
        new_repo_description="Day22 prompt V1: concise factual RAG answer for New SDLC With Vibe Coding.",
    )

    url_v2 = hub.push(
        PROMPT_V2_NAME,
        prompt_v2,
        new_repo_is_public=False,
        new_repo_description="Day22 prompt V2: structured engineering RAG answer for New SDLC With Vibe Coding.",
    )

    print(f"✅ Pushed V1: {PROMPT_V1_NAME}")
    print(f"   {url_v1}")
    print(f"✅ Pushed V2: {PROMPT_V2_NAME}")
    print(f"   {url_v2}")

    return url_v1, url_v2


def pull_prompts_from_hub():
    """
    Pull prompt versions từ LangSmith Prompt Hub.
    """
    print("📥 Pulling prompts from LangSmith Prompt Hub ...")

    prompt_v1 = hub.pull(PROMPT_V1_NAME)
    prompt_v2 = hub.pull(PROMPT_V2_NAME)

    print(f"✅ Pulled V1: {PROMPT_V1_NAME}")
    print(f"✅ Pulled V2: {PROMPT_V2_NAME}")

    return {
        "v1": prompt_v1,
        "v2": prompt_v2,
    }


# ── 3. Deterministic A/B routing ───────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route deterministically bằng MD5 hash.

    Cùng một request_id luôn route về cùng một version.
    """
    digest = hashlib.md5(request_id.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 2
    return "v1" if bucket == 0 else "v2"


# ── 4. Vectorstore + retriever ─────────────────────────────────────────────
def setup_vectorstore():
    embeddings = get_embeddings()

    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=1500, chunk_overlap=250)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore, chunks


def build_retriever(vectorstore, chunks):
    return build_section_anchor_retriever(
        vectorstore=vectorstore,
        chunks=chunks,
        faiss_k=8,
        bm25_k=8,
        weights=(0.35, 0.65),
        max_docs=8,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── 5. A/B RAG query có tracing ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["rag", "step2", "ab-routing", "prompt-hub"])
def ask_ab(
    question: str,
    request_id: str,
    retriever,
    prompts: dict,
    llm,
) -> dict:
    """
    Chạy một câu hỏi qua A/B router:
      request_id → MD5 route → prompt version → retriever → prompt|llm|parser
    """
    version = get_prompt_version(request_id)

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = prompts[version]
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "request_id": request_id,
        "version": version,
        "question": question,
        "answer": answer,
        "num_docs": len(docs),
    }


# ── 6. Logging ─────────────────────────────────────────────────────────────
def write_ab_log(results: list, prompt_urls: tuple):
    evidence_dir = Path(__file__).parent.parent / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    log_path = evidence_dir / "02_ab_routing_log.txt"

    counts = {"v1": 0, "v2": 0}
    for item in results:
        counts[item["version"]] += 1

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Day22 Task 2 — Prompt Hub + A/B Routing Log\n")
        f.write("=" * 72 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Project: {config.LANGSMITH_PROJECT}\n")
        f.write(f"Prompt V1: {PROMPT_V1_NAME}\n")
        f.write(f"Prompt V2: {PROMPT_V2_NAME}\n")
        f.write(f"Prompt V1 URL: {prompt_urls[0]}\n")
        f.write(f"Prompt V2 URL: {prompt_urls[1]}\n")
        f.write(f"Routing method: md5(request_id) % 2\n")
        f.write(f"V1 count: {counts['v1']}\n")
        f.write(f"V2 count: {counts['v2']}\n")
        f.write("=" * 72 + "\n\n")

        for i, item in enumerate(results, 1):
            f.write(f"[{i:02d}] request_id={item['request_id']}\n")
            f.write(f"     version={item['version']}\n")
            f.write(f"     question={item['question']}\n")
            f.write(f"     answer={item['answer']}\n")
            f.write("\n")

    print(f"📝 A/B routing log saved to: {log_path}")
    print(f"📊 Version counts: V1={counts['v1']} | V2={counts['v2']}")

    return log_path, counts


# ── 7. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub + A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    prompt_urls = push_prompts_to_hub()
    prompts = pull_prompts_from_hub()

    vectorstore, chunks = setup_vectorstore()
    retriever = build_retriever(vectorstore, chunks)
    llm = get_llm(temperature=0.0)

    results = []

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"day22-q{i:02d}"

        result = ask_ab(
            question=question,
            request_id=request_id,
            retriever=retriever,
            prompts=prompts,
            llm=llm,
        )

        results.append(result)

        print(
            f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] "
            f"{request_id} → {result['version'].upper()} | "
            f"Q: {question[:70]}"
        )
        print(f"       A: {result['answer'][:300]}\n")

    log_path, counts = write_ab_log(results, prompt_urls)

    if counts["v1"] == 0 or counts["v2"] == 0:
        print("⚠️ Warning: One prompt version received 0 routes. Change request_id strategy.")
    else:
        print("✅ Both prompt versions were used.")

    print("\n✅ Bước 2 hoàn tất.")
    print(f"   Log: {log_path}")
    print("   Mở LangSmith → Prompts để chụp evidence prompt hub.")


if __name__ == "__main__":
    main()