"""
Bước 3 - RAGAS Evaluation
=========================
  1. Chạy 50 QA pairs qua cả hai prompt versions V1 và V2
  2. Tạo EvaluationDataset với SingleTurnSample
  3. Chấm 4 metrics:
     - faithfulness
     - answer_relevancy
     - context_recall
     - context_precision
  4. Lưu report JSON và chart PNG vào evidence/
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import config trước LangChain/LangSmith.
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
    build_section_anchor_retriever,
)
from qa_pairs import QA_PAIRS


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
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{question}"),
        ]
    )


def setup_vectorstore():
    embeddings = get_embeddings()

    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=1500, chunk_overlap=250)
    print(f"Chunks: {len(chunks)}")

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


@traceable(name="ragas-rag-run", tags=["ragas", "step3"])
def run_rag(question: str, prompt, retriever, llm) -> dict:
    """
    Return đúng format cần cho RAGAS:
      answer: str
      contexts: list[str]
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": format_docs(docs),
            "question": question,
        }
    )

    return {
        "answer": answer,
        "contexts": contexts,
    }


def build_samples(limit: int | None = None):
    """
    Chạy QA_PAIRS qua cả V1 và V2, tạo:
      - EvaluationDataset
      - metadata records để join lại sau khi evaluate
    """
    vectorstore, chunks = setup_vectorstore()
    retriever = build_retriever(vectorstore, chunks)
    llm = get_llm(temperature=0.0)

    prompts = {
        "v1": create_prompt(SYSTEM_V1),
        "v2": create_prompt(SYSTEM_V2),
    }

    pairs = QA_PAIRS[:limit] if limit else QA_PAIRS

    samples = []
    records = []

    for i, item in enumerate(pairs, start=1):
        question = item["question"]
        reference = item["reference"]

        for version in ["v1", "v2"]:
            result = run_rag(
                question=question,
                prompt=prompts[version],
                retriever=retriever,
                llm=llm,
            )

            sample = SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=reference,
            )

            samples.append(sample)
            records.append(
                {
                    "question_id": i,
                    "version": version,
                    "question": question,
                    "reference": reference,
                    "answer": result["answer"],
                    "num_contexts": len(result["contexts"]),
                }
            )

            print(
                f"[{i:02d}/{len(pairs)}] {version.upper()} "
                f"question={question[:70]}"
            )
            print(f"       answer={result['answer'][:250]}\n")

    dataset = EvaluationDataset(samples=samples)
    return dataset, records


def run_ragas_evaluation(dataset: EvaluationDataset):
    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    print("Running RAGAS evaluation...")
    print("This can take a long time with local Ollama.")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
        )
    except TypeError:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

    return result


def get_metric_columns(df):
    ignored = {
        "question_id",
        "version",
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
        "question",
        "answer",
        "num_contexts",
    }

    metric_cols = []

    for col in df.columns:
        if col in ignored:
            continue

        try:
            if df[col].dtype.kind in "biufc":
                metric_cols.append(col)
        except Exception:
            pass

    return metric_cols


def save_reports(result, records: list):
    import pandas as pd

    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    evidence_dir = root_dir / "evidence"

    data_dir.mkdir(exist_ok=True)
    evidence_dir.mkdir(exist_ok=True)

    df = result.to_pandas()

    # RAGAS 0.4.3 names this metric by implementation class.
    df = df.rename(
        columns={
            "llm_context_precision_with_reference": "context_precision",
        }
    )

    metadata_df = pd.DataFrame(records)

    for col in ["question_id", "version"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.insert(0, "question_id", metadata_df["question_id"])
    df.insert(1, "version", metadata_df["version"])

    metric_cols = get_metric_columns(df)

    overall_metrics = {
        col: float(df[col].mean())
        for col in metric_cols
    }

    by_version_metrics = {}
    for version in ["v1", "v2"]:
        sub = df[df["version"] == version]
        by_version_metrics[version] = {
            col: float(sub[col].mean())
            for col in metric_cols
        }

    report = {
        "task": "Day22 Task 3 - RAGAS Evaluation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": config.LANGSMITH_PROJECT,
        "num_questions": len(records) // 2,
        "num_samples": len(records),
        "versions": ["v1", "v2"],
        "overall_metrics": overall_metrics,
        "by_version_metrics": by_version_metrics,
        "metric_columns": metric_cols,
        "rows": df.where(pd.notnull(df), None).to_dict(orient="records"),
    }

    data_json_path = data_dir / "ragas_report.json"
    evidence_json_path = evidence_dir / "03_ragas_report.json"
    csv_path = evidence_dir / "03_ragas_rows.csv"
    png_path = evidence_dir / "03_ragas_scores.png"

    data_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    df.to_csv(csv_path, index=False, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        means = df.groupby("version")[metric_cols].mean().T
        ax = means.plot(kind="bar", figsize=(10, 6))
        ax.set_title("RAGAS Scores by Prompt Version")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(title="Prompt Version")
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        print(f"Saved chart: {png_path}")
    except Exception as e:
        print(f"Could not create PNG chart: {type(e).__name__}: {e}")

    print(f"Saved data report: {data_json_path}")
    print(f"Saved evidence report: {evidence_json_path}")
    print(f"Saved rows CSV: {csv_path}")

    print("\nOverall metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nMetrics by version:")
    for version, metrics in by_version_metrics.items():
        print(f"  {version}:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N QA pairs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Build RAGAS samples but do not run RAGAS metrics.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    dataset, records = build_samples(limit=args.limit)

    print(f"Built dataset with {len(records)} samples.")
    print(f"QA pairs used: {len(records) // 2}")

    if args.no_eval:
        print("Skipping RAGAS metrics because --no-eval was provided.")
        return

    result = run_ragas_evaluation(dataset)
    save_reports(result, records)

    print("\nTask 3 completed.")


if __name__ == "__main__":
    main()