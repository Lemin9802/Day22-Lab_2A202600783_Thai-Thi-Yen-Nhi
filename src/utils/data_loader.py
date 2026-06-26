"""
Tiện ích để tải và xử lý dữ liệu PDF cho RAG pipeline.

Day 22 customized version:
- Knowledge base mặc định là data/new_sdlc_vibe_coding.pdf
- Không dùng data/knowledge_base.txt nữa
- Có hỗ trợ hybrid retriever: FAISS + BM25
- Có section-anchor retriever cho các câu hỏi về AI agent components
"""

from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc knowledge base từ file PDF và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file PDF.
              Mặc định: data/new_sdlc_vibe_coding.pdf

    Returns:
        Nội dung PDF dưới dạng str, có đánh dấu số trang.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "new_sdlc_vibe_coding.pdf"

    path = Path(path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF knowledge base is supported. Got: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base PDF not found: {path}")

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(f"\n\n[Page {page_num}]\n{text}")

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError(f"No extractable text found in PDF: {path}")

    print(f"📄 Loaded knowledge base: {path.name}")
    print(f"📚 Pages extracted: {len(pages)}")
    print(f"📝 Characters extracted: {len(full_text)}")

    return full_text


def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 250) -> list:
    """
    Chia văn bản thành các đoạn nhỏ để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    """
    from langchain_community.vectorstores import FAISS

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore


def build_hybrid_retriever(
    vectorstore,
    chunks: list,
    faiss_k: int = 5,
    bm25_k: int = 5,
    weights: tuple = (0.35, 0.65),
):
    """
    Tạo hybrid retriever kết hợp FAISS semantic search và BM25 keyword search.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": faiss_k})

    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = bm25_k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=list(weights),
    )

    return hybrid_retriever


def build_section_anchor_retriever(
    vectorstore,
    chunks: list,
    faiss_k: int = 8,
    bm25_k: int = 8,
    weights: tuple = (0.35, 0.65),
    max_docs: int = 8,
):
    """
    Tạo retriever mạnh hơn cho PDF:
    - Dùng hybrid retrieval FAISS + BM25.
    - Tự động bổ sung đúng section cha cho các câu hỏi dễ bị mơ hồ:
      AI agent basics, agent components, Agent Skills, architecture,
      maintenance, harness, orchestrator skills.

    Đây không hard-code câu trả lời; nó chỉ đưa đúng section liên quan vào context.
    """
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    hybrid_retriever = build_hybrid_retriever(
        vectorstore=vectorstore,
        chunks=chunks,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        weights=weights,
    )

    def contains_any(text: str, terms: list[str]) -> bool:
        return any(term in text for term in terms)

    def query_needs_anchor(query: str, anchor_name: str) -> bool:
        q = query.lower()

        if anchor_name == "agent_basics":
            return (
                "agent" in q
                and contains_any(q, ["what is an ai agent", "what is an agent", "ai agent"])
            )

        if anchor_name == "agent_components":
            return (
                "agent" in q
                and contains_any(
                    q,
                    [
                        "model",
                        "tool",
                        "tools",
                        "memory",
                        "orchestration",
                        "deployment",
                        "five parts",
                        "parts of an ai agent",
                    ],
                )
            )

        if anchor_name == "agent_skills":
            return "agent skills" in q or "skills" in q and "agent" in q

        if anchor_name == "architecture":
            return contains_any(
                q,
                [
                    "architecture",
                    "architectural",
                    "human-centric",
                    "human centric",
                    "architectural decisions",
                ],
            )

        if anchor_name == "maintenance":
            return contains_any(q, ["maintenance", "evolution", "legacy codebase", "technical debt"])

        if anchor_name == "harness":
            return "harness" in q

        if anchor_name == "orchestrator_skills":
            return "orchestrator" in q or "orchestrator mode" in q

        return False

    anchor_phrases = {
        "agent_basics": [
            "an ai agent is a software system that perceives a goal",
            "where a chatbot produces a response and waits for the next prompt",
            "figure 2: the agent loop",
        ],
        "agent_components": [
            "every agent, however simple or sophisticated, is built from five parts",
            "the model is the reasoning engine",
            "tools connect the model to the world",
            "memory is the state",
        ],
        "agent_skills": [
            "the most powerful pattern for managing dynamic context is agent skills",
            "structured, portable packages of procedural knowledge",
            "agent skills have seen rapid adoption",
            "context rot from overloaded prompts",
        ],
        "architecture": [
            "architecture remains the most stubbornly human-centric phase",
            "architectural decisions are fundamentally about trade-offs",
            "ai excels at implementing architectural decisions once they are made",
        ],
        "maintenance": [
            "maintenance and evolution",
            "legacy codebases that were once impenetrable",
            "can now be navigated, understood, and modified with ai assistance",
            "systematically migrate codebases between frameworks",
        ],
        "harness": [
            "what's in the harness",
            "concretely, a harness includes",
            "instructions and rule files",
            "sandboxes and execution environments",
            "observability: logs, traces, evaluations",
        ],
        "orchestrator_skills": [
            "the orchestrator mode requires a different skill set",
            "specification: defining tasks precisely enough",
            "decomposition: breaking large tasks",
            "evaluation: quickly assessing",
            "system design: designing the constraints",
        ],
    }

    def find_anchor_chunks(query: str):
        anchors = []

        for anchor_name, phrases in anchor_phrases.items():
            if not query_needs_anchor(query, anchor_name):
                continue

            for idx, chunk in enumerate(chunks):
                c = chunk.lower()

                if any(phrase in c for phrase in phrases):
                    anchors.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "chunk_id": idx,
                                "source": f"{anchor_name}_anchor",
                            },
                        )
                    )

        return anchors

    def retrieve(query: str):
        raw_docs = hybrid_retriever.invoke(query)

        docs = []
        seen = set()

        for doc in find_anchor_chunks(query):
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                docs.append(doc)

        for doc in raw_docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                docs.append(doc)

        return docs[:max_docs]

    return RunnableLambda(retrieve)