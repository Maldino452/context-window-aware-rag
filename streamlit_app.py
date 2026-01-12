import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from vector_db import retriever
import context
import os


st.set_page_config(
    page_title="T&E Policy Assistant",
    page_icon="✈️",
    layout="wide", 
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: white;
        color: #333;
        padding: 10px;
        text-align: left;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #f5f5f5;
        border-color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():

    LLM_MODEL = "llama3.2:1b"
    
    try:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
        return llm, "ready"
    except Exception as e:
        return None, None, str(e)

model, status = load_system()


def display_token_breakdown(breakdown, total_tokens, overflow_occurred):
    """Display token budget breakdown in table format like terminal output."""
    
    st.markdown("### 📊 Context Budget Breakdown")
    
    sections = ['instructions', 'goal', 'memory', 'retrieval', 'tool_outputs']
    
    tokens_row = []
    percentage_row = []
    source_row = []
    
    for section in sections:
        data = breakdown[section]
        used = data['tokens_used']
        budget = data['budget']
        percentage = (used / budget) * 100
        source = data['source']
        
        tokens_row.append(f"{used}/{budget}")
        
        if data['truncated']:
            percentage_row.append("⚠ TRUNCATED")
        else:
            percentage_row.append(f"{percentage:.0f}%")
        
        source_row.append(source)
    
    import pandas as pd
    table_data = pd.DataFrame({
        'Instructions': [tokens_row[0], percentage_row[0], source_row[0]],
        'Goal': [tokens_row[1], percentage_row[1], source_row[1]],
        'Memory': [tokens_row[2], percentage_row[2], source_row[2]],
        'Retrieval': [tokens_row[3], percentage_row[3], source_row[3]],
        'Tool Outputs': [tokens_row[4], percentage_row[4], source_row[4]]
    }, index=['Tokens', 'Status', 'Source'])
    
    st.dataframe(
        table_data, 
        use_container_width=True,
        height=150
    )
    
    if overflow_occurred:
        ret_data = breakdown['retrieval']
        if ret_data['truncated']:
            st.warning(f"⚠️ **Budget Overflow:** Kept {ret_data.get('chunks_kept', 0)} chunks, dropped {ret_data.get('chunks_dropped', 0)} chunks (Original: {ret_data.get('original_tokens', 'N/A')} tokens)")
    
    st.caption(f"**Total Context:** {total_tokens} tokens")


with st.sidebar:
    st.title("T&E Policy Assistant")
    
    if status == "ready":
        st.success("● System Ready")
    else:
        st.error(f"● Error: {status}")
        if status == "Database not found":
            st.info("Run: `python vector_db.py` first")
        st.stop()
    
    st.divider()
    
    st.subheader("Display Options")
    show_breakdown = st.toggle("Show token breakdown", value=True)
    show_context = st.toggle("Show assembled context", value=False)
    show_sources = st.toggle("Show source chunks", value=False)
    
    st.divider()
    
    with st.expander("📊 Token Budgets"):
        budgets = context.BUDGETS
        st.markdown(f"""
        - **Instructions:** {budgets['instructions']} tokens
        - **Goal:** {budgets['goal']} tokens  
        - **Memory:** {budgets['memory']} tokens
        - **Retrieval:** {budgets['retrieval']} tokens
        - **Tool Outputs:** {budgets['tool_outputs']} tokens
        """)
    
    st.divider()
    
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Context-Window-Aware RAG**
        
        Demonstrates explicit token budget management.
        
        When retrieval exceeds 550 tokens, the system prioritizes most relevant chunks.
        """)
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.clear()
        st.rerun()


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


st.title("✈️ T&E Policy Assistant")
st.caption("*Aurelius Consulting Group | Context-Aware RAG System*")

st.divider()


with st.expander("💡 Example Questions", expanded=True):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Simple Queries**")
        if st.button("🧾 Receipt rules?"):
            st.session_state.current_query = "When do I need receipts for expenses?"
        if st.button("🍽️ Meal allowance?"):
            st.session_state.current_query = "What's the meal allowance for domestic travel?"
        if st.button("💼 Can I expense Uber?"):
            st.session_state.current_query = "Can I expense Uber rides during business travel?"
    
    with col2:
        st.markdown("**Complex Queries**")
        if st.button("🌍 All international rules?"):
            st.session_state.current_query = "What are all the rules for international travel including flights, hotels, and meals?"
        if st.button("🇬🇧 Travel to London?"):
            st.session_state.current_query = "I'm traveling to London next week - what do I need to know?"
        if st.button("🚗 All transport policies?"):
            st.session_state.current_query = "Tell me about ground transportation including Uber, taxis, and rental cars"


for message in st.session_state.messages:
    if message["role"] == "user":
        cols = st.columns([3, 1])
        with cols[1]:
            st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: right;'>👤</div>", unsafe_allow_html=True)
    
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            
            if "breakdown" in message:
                if show_breakdown:
                    st.divider()
                    display_token_breakdown(message["breakdown"], message["total_tokens"], message["overflow"])



if 'current_query' in st.session_state:
    user_input = st.session_state.current_query
    del st.session_state.current_query
else:
    user_input = st.chat_input("Ask about T&E policies...")



if user_input:
    
    cols = st.columns([3, 1])
    with cols[1]:
        st.markdown(f"<div style='text-align: right;'>{user_input} 👤</div>", unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        
        with st.status("🔍 Processing...", expanded=False) as status:
            
            st.write("Searching policies...")
            retrieved_docs = retriever.invoke(user_input)
            st.write(f"✓ Retrieved {len(retrieved_docs)} chunks")
            
            st.write("Assembling context with token budgets...")
            assembled_context, breakdown, overflow_occurred, total_tokens = context.assemble_context(
                user_question=user_input,
                retrieved_docs=retrieved_docs,
                conversation_history=st.session_state.conversation_history,
                memory_items=None,
                tool_results=None
            )
            st.write("✓ Context assembled")
            
            st.write("Generating answer...")
            prompt = f"""{assembled_context}

            ---

            Question: {user_input}

            Answer (be concise and cite relevant policies):
            """
            
            answer = model.invoke(prompt)
            st.write("✓ Complete")
            
            status.update(label="✅ Done!", state="complete")
        
        st.markdown(answer)
        
        if show_breakdown:
            st.divider()
            display_token_breakdown(breakdown, total_tokens, overflow_occurred)
        
        if show_context:
            with st.expander("📄 Assembled Context"):
                st.code(assembled_context, language="text")
        
        if show_sources:
            st.divider()
            st.markdown("### 📚 Source Chunks")
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"Chunk {i}: {doc.metadata.get('source', 'unknown')}"):
                    st.text(doc.page_content)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "breakdown": breakdown,
        "total_tokens": total_tokens,
        "overflow": overflow_occurred
    })
    
    st.session_state.conversation_history.append({
        'role': 'user',
        'content': user_input
    })
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': answer[:200]
    })
    
    if len(st.session_state.conversation_history) > 6:
        st.session_state.conversation_history = st.session_state.conversation_history[-6:]

