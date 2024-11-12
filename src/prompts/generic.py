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

# prompts for response/reward
prompt_rating = "Instruction: Determine whether the provided context is relevant to the given query? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating. Rate 0 if the context is empty."
guideline = "Guideline:\n- 5: The context is highly relevant, complete, and accurate to the query.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the query.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the query.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the query.\n- 1: The context is minimally relevant or complete, with substantial shortcomings to the query.\n- 0: The context is not relevant or complete at all."
template_rating = "{prompt_rating}\n\n{guideline}\n\nQuery: {Q}\n\nContext: {D}\n\nRating:\n"

def apply_rsp_inst_prompt(Q, D, A="", prefix="Rating:\n"):
    p = template_rating.replace("{prompt_rating}", prompt_rating).replace("{guideline}", guideline)
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    return p

# prompts for feedback
# prompt_report = "Instruction: Write an accurate, engaging, and concise report for the given topic using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
# template_report = f"{prompt_report}\n\nTopic: {Q}\n\nSearch results:\n{D}\nReport:\n"
prompt_report = "Instruction: Explain the information need of the question in detail. You may find the useful information in the given texts (some of which might be irrelevant). Write the explanation witin 50 words."
template_report = "{prompt_report}\n\nQuestion: {Q}\n\nTexts:\n{D}\nExplanation:\n"
# prompt_report = "Instruction: Rewrite the question based on the useful information in given texts (some of which might be irrelevant). Rewrite a new concise question with more clear information need."
# template_report = "{prompt_report}\n\nQuestion: {Q}\n\nTexts:\n{D}\nRewritten question:\n"

def apply_fbk_inst_prompt(Q, D, prefix="Report:\n", *kwargs):
    p = template_report.replace('{prompt_report}', prompt_report)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p
