INSTR = ("Instruction: Determine if clarification is needed based on the following multi-agent context and output JSON."
         "Output strictly as {\"need_clarify\":<true or false>,"
         "\"to_agent\":<null or string>,\"type\":<null or DG|RD|SC|CG>,"
         "\"clarify_question\":<null or string>}."
         "When no clarification is needed, need_clarify=false and others are null.")

def build_input(example):
    query = example.get('query', '')
    from_agent = example.get('from_agent', '')  
    to_agent = example.get('to_agent', '')    
    original_response = example.get('original_response', '') 
    new_response = example.get('new_response', '')
    
    
    context_parts = [
        f"query: {query}",
        f"from_agent: {from_agent}",
    ]
    
    if to_agent:
        context_parts.append(f"to_agent: {to_agent}")
    
    context_parts.extend([
        f"original_response: {original_response}",
        f"new_response: {new_response}" if new_response else "new_response: "
    ])
    
    return (
        f"{INSTR}\n\nContext:\n" +
        "\n".join(context_parts) +
        "\OUTPUT:"
    )

def json_or_null(v):
    if v is None: return "null"
    return '"' + str(v).replace('"','\\"') + '"'

def _json_bool_or_null(v):
    if v is None: return "null"
    return "true" if bool(v) else "false"

def target_json_str(target):
    return (
        '{'
        f"\"need_clarify\": {_json_bool_or_null(target.get('need_clarify'))}, "
        f"\"to_agent\": {json_or_null(target.get('to_agent'))}, "
        f"\"type\": {json_or_null(target.get('type'))}, "
        f"\"clarify_question\": {json_or_null(target.get('clarify_question'))}"
        '}'
    )
