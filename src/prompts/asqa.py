# prompts for answering
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch result:\n{D}\n{PREFIX}"
instruction_prompt = "Answer the question using the provided search result (some of which might be irrelevant)."
doc_prompt_template = "[{ID}] (Title: {T}) {P}\n"

inst_prompt_template_new = "{INST}\n\nQuestion: {Q}\nSuggested answer: {A}\n\nSearch result:\n{D}\n{PREFIX}"
instruction_prompt_new = "Determine whether the following search result can answer the given question. A suggested answer is also provided for reference. Rate the search result as 0, 0.5, or 1, based on the guidelines below. Provide only the rating.\n\nGuideline:\n- 1: The search result fully answers the question and matches the relevant details in the suggested answer.\n- 0.5: The search result can answer the question but lacks sufficient detail.\n- 0: The search result cannot adequately answer the question."
instruction_prompt_new = "Determine how useful the following search result can help answering the given question. A suggested answer is also provided for reference. Rate the search result as 0 to 1, based on the guidelines below. Provide only the rating.\n\nGuideline:\n- 1 means the search result fully answers the question and matches the relevant details in the suggested answer.\n- 0 means the search result cannot adequately answer the question."

## ===== WITH ANSWER =====
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nInitial search resulte\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Read the given question and review the initial search result listed below. Identify the relevance information and provide a short feedback that can help search engine find more useful documents. Write the feedback within `<f>` and `</f>` tags."
# fbk_instruction_prompt = "Read the provided question and review the initial search result listed below. Identify any possible missing or incomplete information, and suggest an appropriate follow-up query in needed. Output empty string if the initial search result is sufficiently clear. Write the follow-up query within `<q>` and `</q>` tags."

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

def apply_fbk_inst_prompt(Q, D, instruction="", prefix="Keywords:\n", *kwargs):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
