instruction_prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided documents as references (some of which might be irrelevant). Always cite at least one document for every sentences in the answer. Use the citation format of square brackets to indicate the cited documents (e.g., [1] for the first reference). If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Only generate the answer, excluding any disclaimers, notes or list of references at the end of the answer."

demo_sep = "\n\n\n"
# doc_prompt_template = "Document [{ID}]: (Title: {T}) {P}\n"
doc_prompt_template = "Document: (Title: {T}) {P}\n"
demo_prompt_template = "{INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
inst_prompt_template = "{INST}\n\n{DEMO}Question: {Q}\n\n{D}\nAnswer: {A}"

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
