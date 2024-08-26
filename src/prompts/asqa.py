# prompts for answering
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch results:\n{D}\n{PREFIX}"
instruction_prompt = "Answer the question using the provided search results (some of which might be irrelevant)."
# doc_prompt_template = "[{ID}] (Title: {T}) {P}\n"
doc_prompt_template = "[{ID}] {P}\n"
# doc_prompt_template = "- {P}\n"

## ===== WITH ANSWER =====
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nAnswer: {A}\n\nInitial search results:\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Read the given question and examine the example answer. The answer was generated using the initial search results listed below. Please provide additional keywords that can obtain more helpful search results in terms of correctness and comprehensiveness."

def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item.get('title', ''))
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_rsp_inst_prompt(Q, D, instruction="", prefix="Answer: "):
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
