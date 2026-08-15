"""
Document Q&A Pipeline — YOUR WORK GOES HERE.

The knowledge base (loading, chunking, vector store) is already built
for you in knowledge_base.py. Your job is to:

  1. Retrieve relevant chunks and generate an answer
  2. Wire it up into an interactive CLI

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - HuggingFace pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from knowledge_base import build_knowledge_base

from langchain_community.vectorstores import FAISS


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store: FAISS, llm, question: str) -> dict[str, str]:
    """Retrieve relevant chunks and generate an answer.

    Steps:
      1. Use vector_store.similarity_search(question, k=3) to get
         the top 3 most relevant document chunks.
      2. Combine the chunk text into a single context string.
         (Hint: each chunk has a .page_content attribute)
      3. Format the PROMPT_TEMPLATE with the context and question.
      4. Pass the formatted prompt to llm(...) and extract the
         generated text from the result.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    # TODO: implement this (~6-8 lines)
    # raise NotImplementedError("TODO 1: Implement ask_question")
    similarity_ans = vector_store.similarity_search(question, k=3)
    context: str = ""
    text_chunks: list[str] = []
    for doc in similarity_ans:
        text_chunks.append(doc.page_content)
        context += doc.page_content

    return {
        "answer": llm(PROMPT_TEMPLATE.format(context=context, question=question))[0]["generated_text"],
        "sources": text_chunks
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """Interactive Q&A loop.

    Steps:
      1. Build the knowledge base using build_knowledge_base()
         with the data/ directory path.
      2. Load the LLM using get_llm().
      3. Start a loop that:
         - Prompts the user for a question with input()
         - Exits if they type "quit"
         - Calls ask_question() with their input
         - Prints the retrieved sources and the answer
    """
    # TODO: implement this (~10-12 lines)

    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    knowledge_base_vec: FAISS = build_knowledge_base(data_dir=data_dir)
    while True:
        question: str = input("> ")

        if question.lower() == "quit":
            break
        if question:  # Checks if input is empty
            # Input response will split on the CLI command passed in for single-question mode
            received_input: list[str] | str = question.split("--") if question.endswith("--query") else question
            if isinstance(received_input, str):
                answer: dict[str, str] = ask_question(vector_store=knowledge_base_vec, llm=get_llm(), question=received_input)
            else:
                answer: dict[str, str] = ask_question(vector_store=knowledge_base_vec, llm=get_llm(), question=received_input[0])
            
            print("Sources:")
            [print(f"  {i + 1}. {source.replace("\n", " - ")}") for i, source in enumerate(answer['sources'])]
            print(f"\nAnswer: {answer["answer"]}")
            if isinstance(received_input, list):
                if "query" in received_input[1]:
                    break
        else:
            print("Please ask a question to retrieve a response (e.g. 'How much does the Growth package cost?')")


if __name__ == "__main__":
    main()