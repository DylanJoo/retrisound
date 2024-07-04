# prompts for answering
instruction_prompt = "Answer the given question using the provided documents as references (some of which might be irrelevant). The answer should be as comprehensive as possible."
doc_prompt_template = "Document [{ID}]: (Title: {T}) {P}\n"
demo_prompt_template = "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
inst_prompt_template = "{INST}\n\n{DEMO}Question: {Q}\n\n{D}\nAnswer: {A}"
demo_sep = "\n\n"

# promtps for asking feedback
## with answer 
# TBD

## without answer 
### prompt at once
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Read and understand the given question. Then, based on the question, identify the useful information in the provided search results (some of which might be irrelevant)."
# fbk_prefix="Rewritten question: "
# fbk_instruction_prompt += " Finally, rewrite the question for searching additional new documents. These new documents are expected to complete the missing knowledge about the question." 
# fbk_prefix="New keyword combintation:"
# fbk_instruction_prompt += " Finally, write a new keyword combintation for searching additional new documents. These new documents are expected to complete the missing knowledge about the question." 

### [DEBUG] other options
# Write the more important keywords at the beginning of the list, vice versa.

### prompt separately
fbk_prefix="Based on the identified useful information, rewrite the question for searching additional additional new documents. These new documents are expected to complete the missing knowledge about the question.\n\nRewritten question:"
# fbk_prefix="Based the identified useful information, write a new new keyword combination for searching additional new documents. These new documents are expected to complete the missing knowledge about the question.\n\nNew keyword combination:"

def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item.get('title', ''))
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_demo_prompt(Q, D, A, instruction=""):
    p = demo_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    return p

def apply_inst_prompt(Q, D, instruction="", add_prefix=True):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", "")
    if add_prefix is False:
        p = p.replace("Answer:", "").strip()
    return p

def apply_fbk_inst_prompt(Q, D, instruction="", prefix=""):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
