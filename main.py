from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from vector_db import retriever
import context



LLM_MODEL = "llama3.2:1b"

model = OllamaLLM(model=LLM_MODEL, temperature=0.1)


# STRORE CONVERSATION HISTORY AND MEMORY ITEMS
conversation_history = []
memory_items = []



while True:
    print("\n" + "-"*70)
    
    question = input("\n📝 Your question: ").strip()
    
    if not question:
        continue
    
    if question.lower() in ['quit', 'exit']:
        print("\n👋 Goodbye!\n")
        break

    
    print("\n" + "="*70)
    
    # WE RETRIEVE RELEVANT DOCUMENTS/CHUNKS BASED ON QUESTION
    retrieved_docs = retriever.invoke(question)
    
    print(f"   ✓ Retrieved {len(retrieved_docs)} relevant chunks")
    
    # WE THEN ASSEMBLE THE CONTEXT WITH THE BUDGET
    assembled_context, breakdown, overflow_occurred, total_tokens = context.assemble_context(
        user_question=question,
        retrieved_docs=retrieved_docs,
        conversation_history=conversation_history,
        memory_items=memory_items,
        tool_results=None
    )
    
    context.display_breakdown(breakdown, total_tokens)
    
    
    # WE CREATE THE FINAL PROMPT USING THE QUESTION PLUS THE RETRIVED RELEVANT CHUNKS AND WE SEND THAT TO THE LLM TO GET AN ANSWER
    prompt = f"""{assembled_context}

    ---

    Question: {question}

    Answer (be concise and cite relevant policies):
    """
    
    answer = model.invoke(prompt)
    
    
    print("\n" + "="*70)
    print("💡 ANSWER")
    print("="*70)
    print(answer)
    print("="*70)
    
    # WE ADD TO THE CONVERSATION HISTORY
    conversation_history.append({
        'role': 'user',
        'content': question
    })
    conversation_history.append({
        'role': 'assistant',
        'content': answer[:200]
    })
    
    # WE KEEP HISTORY MANAGEABLE BY STORING ONLY THE MOST RECENT TURNS
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]
    
