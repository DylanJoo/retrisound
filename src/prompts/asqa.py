# prompts for answering
instruction_prompt = "Answer the given question using the provided search results (some of which might be irrelevant)."
doc_prompt_template = "[{ID}]: (Title: {T}) {P}\n"
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nOld search results:\n{D}\n{PREFIX}"

## ===== WITH ANSWER =====
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q} Example answer: {A}\n\nOld search results:\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Read and examine the given question and the example asnwer. The example answer is generated according to the given old search results. Write a new query for new search results. The new search results should help answer the question more comprehensively comparing to the older search results."
fbk_prefix="New query:"

## ===== WITHOUT ANSWER =====
### (2) Instruction first + Query expansion
# fbk_instruction_prompt += " Finally, write a new keyword combintation for searching additional new documents. These new documents are expected to complete the missing knowledge about the question." 
# fbk_prefix="New keyword combintation:"

## [Two-part instructions] One for identifying gap, one for reasoning the task. romtps for asking feedback
## (3) Instruction later + Query rewriting
# fbk_instruction_prompt = "Read and understand the given question. Then, based on the question, identify the useful information in the provided search results (some of which might be irrelevant)."
# fbk_prefix="In addition to the provided search results, rewrite a next question for next searching. These additional new documents are expected to complete the missing knowledge about the question.\n\nRewritten next question:"

### (4) Instruction later + Query expansion
# fbk_instruction_prompt = "Read and understand the given question. Then, based on the question, identify the useful information in the provided search results (some of which might be irrelevant)."
# fbk_prefix="Based the identified useful information, write a new new keyword combination for searching additional new documents. These new documents are expected to complete the missing knowledge about the question.\n\nNew keyword combination: "

## with answer 
# TBD

def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item.get('title', ''))
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_inst_prompt(Q, D, instruction="", prefix=""):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", "")
    p = p.replace("{PREFIX}", prefix).strip()
    return p

def apply_fbk_inst_prompt(Q, D, instruction="", prefix=""):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
