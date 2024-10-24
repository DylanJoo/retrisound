# context template
doc_prompt_template = "[{ID}]{T}{P}\n"
def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        title = doc_item.get('title', '')
        if title != '':
            p_doc = p_doc.replace("{T}", f" (Title: {title}) ")
        else:
            p_doc = p_doc.replace("{T}", "")
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

# prompts for response
inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch results:\n{D}\n{PREFIX}"
inst_prompt_template = "{INST}\n\nQuestion: {Q}\nSuggested answer: {A}\n\nSearch result:\n{D}\n{PREFIX}"
instruction_prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
instruction_prompt = "Instruction: Determine how many useful the provided search results can help answering the given question. A human written suggested answer is also provided for your reference. Rate the search results from 0 to the number of documents. Only provide one overall rating for all of the documents.\n\nGuideline:\n- 2: The search results completely answer the question and match the details of the suggested answer.\n- 1: The search results can answer the question but lack some details of the suggested answer.\n- 0: The search results cannot answer the question."

def apply_rsp_inst_prompt(Q, D, A, instruction="", prefix="Rating:\n"):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    p = p.replace("{PREFIX}", prefix).strip()
    return p

inst_prompt_template = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nSuggested answer: {A}\n\nContext: {D}\n\n{PREFIX}" 
guideline = "- 5: The context is highly relevant, complete, and accurate.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The context is minimally relevant or complete, with substantial shortcomings.\n- 0: The context is not relevant or complete at all."
instruction_prompt = "Determine whether the question can be answered based on the provided context? The suggested answer is also provided for reference. Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating. Rate 0 if the context is empty."
def apply_rsp_inst_prompt(Q, D, A, instruction="", prefix="Rating:"):
    p = inst_prompt_template
    p = p.replace("{G}", guideline)
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{D}", D)
    p = p.replace("{A}", A)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
    # p_output = []
    # for d in D:
    #     pi = p.replace("{d}", d)
    #     p_output.append(pi)
    # return p_output

# prompts for feedback
fbk_inst_prompt_template = "{INST}\n\nQuestion: {Q}\n\nSearch results:\n{D}\n{PREFIX}"
fbk_instruction_prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."

def apply_fbk_inst_prompt(Q, D, instruction="", prefix="Report:\n", *kwargs):
    p = fbk_inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D)
    p = p.replace("{PREFIX}", prefix).strip()
    return p
