import tiktoken


BUDGETS = {
    'instructions': 255,
    'goal': 1500,
    'memory': 55,
    'retrieval': 550,
    'tool_outputs': 855
}



def count_tokens(text):

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# WE COUNT THE CURRENT TOKENS, AND IF THEY EXCEED THE MAXIMUM NUMBER OF TOKENS ALLOWED, WE DISCARD ALL TOKENS AFTER THE MAX NUMBER OF TOKENS
# WE THEN RETURN (text, number of tokens used, T/F - if text was truncated)
def truncate_to_budget(text, max_tokens):
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text, len(tokens), False
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    
    return truncated_text, max_tokens, True



# INSTRUCTIONS WHICH TELLS THE AI AGENT HOW TO BEHAVE 
# MUST TRUNCATE IF THEY EXCEED THE 255 MAXIMUM TOKEN CONSTRAINT
def build_instructions():

    instructions = """
    You are a helpful Travel & Expense Policy Assistant for Aurelius Consulting Group.

    Your role:
    - Answer questions about Travel and Expense policies clearly and accurately
    - Cite specific policy sections when relevant
    - If a policy doesn't cover something, say so clearly
    - Be concise but complete in your responses
    - Use a friendly, professional tone

    Always base your answers on the provided policy documents. Do not make up policies.
    """
    
    tokens = count_tokens(instructions)
    budget = BUDGETS['instructions']
    
    if tokens > budget:
        instructions, tokens, truncated = truncate_to_budget(instructions, budget)
    else:
        truncated = False
    
    return {
        'content': instructions,
        'tokens_used': tokens,
        'budget': budget,
        'truncated': truncated,
        'source': 'System prompt'
    }


# THE USER'S CURRENT QUESTION AND RECENT CONVERSATION CONTEXT
# IF OVER 1500 TOKENS, TRUNCATE AND KEEP ONLY THE CURRENT QUESTION
def build_goal(user_question, conversation_history=None):

    goal = f"Current Question: {user_question}\n\n"
    
    if conversation_history and len(conversation_history) > 0:
        goal += "Recent Conversation:\n"
        for turn in conversation_history[-3:]:
            role = turn.get('role', 'user')
            content = turn.get('content', '')[:200]
            goal += f"{role}: {content}\n"
    
    tokens = count_tokens(goal)
    budget = BUDGETS['goal']
    
    if tokens > budget:
        goal = f"Current Question: {user_question}"
        tokens = count_tokens(goal)
        truncated = True
    else:
        truncated = False
    
    return {
        'content': goal,
        'tokens_used': tokens,
        'budget': budget,
        'truncated': truncated,
        'source': 'User input + conversation history'
    }


# STORED CONVERSATION FACTS
# IF EXCEED THE 55 TOKEN LIMIT, KEEP ONLY THE LATEST 2 TO 3 KEY FACTS
def build_memory(memory_items=None):

    if not memory_items or len(memory_items) == 0:
        memory = "No prior conversation context."
    else:
        memory = "\n".join(memory_items)
    
    tokens = count_tokens(memory)
    budget = BUDGETS['memory']
    
    if tokens > budget:
        memory = "\n".join(memory_items[-2:])
        tokens = count_tokens(memory)
        
        if tokens > budget:
            memory, tokens, _ = truncate_to_budget(memory, budget)
        
        truncated = True
    else:
        truncated = False
    
    return {
        'content': memory,
        'tokens_used': tokens,
        'budget': budget,
        'truncated': truncated,
        'source': 'Conversation memory store'
    }


# VECTOR DATABASE RETRIEVAL RESULTS
# KEEP CHUNKS IN ORDER OF SIMILARITY SCORES
# WHEN OVER 550 TOKENS THEN DROP LOWER RELEVANCE CHUNKS AND RETAIN THE TOP 2-3 MOST RELEVANT CHUNKS
def build_retrieval(retrieved_docs):

    if not retrieved_docs or len(retrieved_docs) == 0:
        return {
            'content': "No relevant policy documents found.",
            'tokens_used': 7,
            'budget': BUDGETS['retrieval'],
            'truncated': False,
            'source': 'Vector database',
            'chunks_kept': 0,
            'chunks_dropped': 0
        }
    
    budget = BUDGETS['retrieval']
    
    retrieval_text = "=== RELEVANT POLICY SECTIONS ===\n\n"
    current_tokens = count_tokens(retrieval_text)
    
    chunks_kept = 0
    chunks_dropped = 0
    original_tokens = current_tokens
    
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content
        
        chunk_text = f"[Source {i}: {source}]\n{content}\n\n"
        chunk_tokens = count_tokens(chunk_text)
        
        total_if_added = current_tokens + chunk_tokens
        
        if total_if_added <= budget:
            retrieval_text += chunk_text
            current_tokens = total_if_added
            chunks_kept += 1
        else:
            remaining_budget = budget - current_tokens
            
            if remaining_budget > 100 and chunks_kept < 2:
                partial_content, _, _ = truncate_to_budget(content, remaining_budget - 50)
                retrieval_text += f"[Source {i}: {source}]\n{partial_content}...[TRUNCATED]\n\n"
                current_tokens = budget
                chunks_kept += 1
                chunks_dropped = len(retrieved_docs) - chunks_kept
                break
            else:
                chunks_dropped += 1
    
    original_tokens = sum(count_tokens(doc.page_content) for doc in retrieved_docs)
    truncated = (chunks_dropped > 0)
    
    return {
        'content': retrieval_text,
        'tokens_used': current_tokens,
        'budget': budget,
        'truncated': truncated,
        'source': 'Vector database retrieval',
        'chunks_kept': chunks_kept,
        'chunks_dropped': chunks_dropped,
        'original_tokens': original_tokens
    }


# TOOL EXECUTION HISTORY
# IF EXCEEDS THE 855 TOKENS LIMIT, THEN TRUNCATE OLDEST RESULTS
def build_tool_outputs(tool_results=None):

    if not tool_results or len(tool_results) == 0:
        tool_outputs = "No recent tool outputs."
    else:
        tool_outputs = "\n\n".join(tool_results[-3:]) 
    
    tokens = count_tokens(tool_outputs)
    budget = BUDGETS['tool_outputs']
    
    if tokens > budget:
        tool_outputs, tokens, _ = truncate_to_budget(tool_outputs, budget)
        truncated = True
    else:
        truncated = False
    
    return {
        'content': tool_outputs,
        'tokens_used': tokens,
        'budget': budget,
        'truncated': truncated,
        'source': 'Tool execution history'
    }


# WE ASSEMBLE THE CONTEXT WITH ALL THE 5 SECTIONS (instructions, goal, memory, retrieval, recent tool outputs)
# AND RETURN THE FULL CONTEXT STRING TO SEND TO LLM, DETAILED TOKEN USAGE FOR EACH SECTION, AND BOOLEAN TO INDICATE IF ANY TRUNCATION TOOK PLACE 
def assemble_context(user_question, retrieved_docs, conversation_history=None, memory_items=None, tool_results=None):

    instructions = build_instructions()
    goal = build_goal(user_question, conversation_history)
    memory = build_memory(memory_items)
    retrieval = build_retrieval(retrieved_docs)
    tool_outputs = build_tool_outputs(tool_results)
    
    assembled = f"""
    {instructions['content']}

    ---

    {goal['content']}

    ---

    Memory:
    {memory['content']}

    ---

    {retrieval['content']}

    ---

    Tool Outputs:
    {tool_outputs['content']}
    """
            
    breakdown = {
        'instructions': instructions,
        'goal': goal,
        'memory': memory,
        'retrieval': retrieval,
        'tool_outputs': tool_outputs
    }
    
    overflow_occurred = any(section['truncated'] for section in breakdown.values())
    
    total_tokens = sum(section['tokens_used'] for section in breakdown.values())
    
    return assembled, breakdown, overflow_occurred, total_tokens



def display_breakdown(breakdown, total_tokens):

    print("\n" + "="*70)
    print("CONTEXT BUDGET BREAKDOWN")
    print("="*70)
    
    for section_name, data in breakdown.items():
        used = data['tokens_used']
        budget = data['budget']
        percentage = (used / budget) * 100
        
        if data['truncated']:
            status = "⚠ TRUNCATED"
        elif percentage > 90:
            status = "⚠ NEAR LIMIT"
        else:
            status = "✓ OK"
        
        print(f"\n{section_name.upper()}: {used}/{budget} tokens ({percentage:.0f}%) {status}")
        print(f"  Source: {data['source']}")
        
        if data['truncated']:
            if section_name == 'retrieval' and 'chunks_dropped' in data:
                print(f"  → Kept {data['chunks_kept']} chunks, dropped {data['chunks_dropped']} chunks")
                print(f"  → Original retrieval: {data['original_tokens']} tokens")
            else:
                print(f"  → Content was truncated to fit budget")
    
    print("\n" + "="*70)
    print(f"TOTAL CONTEXT: {total_tokens} tokens")
    print("="*70)