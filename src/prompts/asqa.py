# prompts for answering
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch results:\n{D}\n{PREFIX}"
instruction_prompt = "Answer the question using the provided search results (some of which might be irrelevant)."
doc_prompt_template = "[{ID}] (Title: {T}) {P}\n"

inst_prompt_template_new = "{INST}\n\nQuestion: {Q}\nSuggested answer: {A}\n\nSearch result:\n{D}\n{PREFIX}"

instruction_prompt_new = "Determine whether the question can be answered based on the following search result? (the suggested answer to the question is also provided as reference). Rate the search result with on a scale from 0 to 3 according to the guideline below. Do not write anything except the rating.\n\nGuideline:\n- 3: The search result can be used to answer the question completely and accurately.\n- 2: The seach result can be used to answer the question partially but missing some details. \n- 1: The search result can not be used to answer the question but has some relevant information to the question.\n- 0: The search result can not be used to answer the question and there are non relevant at all."

## ===== WITH ANSWER =====
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\nExample answer: {A}\n\nInitial search results:\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Read the given question and examine the example answer. The answer was generated using the initial search results listed below. Please provide additional keywords that will help search engine return more helpful documents in terms of correctness and comprehensiveness. Output an empty string if the initial search results are good enougth for answering."

def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item.get('title', ''))
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_rsp_inst_prompt_new(Q, D, A, instruction="", prefix="Rating:\n"):
    p = inst_prompt_template_new
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    p = p.replace("{PREFIX}", prefix).strip()
    return p

def apply_rsp_inst_prompt(Q, D, instruction="", prefix="Answer:\n", **kwargs):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p

def apply_fbk_inst_prompt(Q, D, A="no answer.", instruction="", prefix="Keywords: "):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
