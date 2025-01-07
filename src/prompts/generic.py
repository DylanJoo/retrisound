# context template
doc_prompt_template = "[{ID}]{T}{P}\n"
doc_prompt_template = "{T}{P}\n"
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
# prompt_rating = "Instruction: Determine whether the provided context is relevant to the given query? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating. Rate 0 if the context is empty."
# guideline = "Guideline:\n- 5: The context is highly relevant, complete, and accurate to the query.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the query.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the query.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the query.\n- 1: The context is minimally relevant or complete, with substantial shortcomings to the query.\n- 0: The context is not relevant or complete at all."
# prompt_rating = "Determine whether the provided context is relevant to the given query? Rate the context with on a scale of 0 or 1 according to the guideline below. Do not write anything except the rating. Rate 0 if the context is empty."
# guideline = "Guideline:\n- 1: The context is highly relevant, complete, and accurate to the query.\n- 0: The context is not relevant or incomplete, with substantial shortcomings to the query."
# template_rating = "{prompt_rating}\n\n{guideline}\n\nQuery: {Q}\n\nContext: {D}\n\nRating:\n"
#
# def apply_rsp_inst_prompt(Q, D, A="", prefix="Rating:\n"):
#     p = template_rating.replace("{prompt_rating}", prompt_rating).replace("{guideline}", guideline)
#     p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
#     return p

# prompts for feedback
prompt_report = "Write a concise report about the topic of the given question. Use the provided search results (ignore the irrelevant result) to draft a report within 200 words."
template_report = "{prompt_report}\n\nQueestion: {Q}\n\nSearch results:\n{D}\nReport:\n "

# prompt_report = "Write an accurate, engaging, and concise report for the given topic. Use only the provided search results (some of which might be irrelevant) and cite them properly. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence."
# template_report = "{prompt_report}\n\nTopic: {Q}\n\nSearch results:\n{D}\nReport:\n:"

# prompt_report = "Elaborate the information need of the question in detail. Find the useful information in the given contexts (some of which might be irrelevant, please ignore). Write the explanation witin 50 words."
# template_report = "{prompt_report}\n\nQuestion: {Q}\n\nContexts:\n{D}\nExplanation:\n"

# prompt_report = "Rewrite the question with more comprehensive contexts, making the question easier to understand. Some useful preliminary knowledge could be found in the given texts (but some of which might be irrelevant)."
# template_report = "{prompt_report}\n\nQuestion: {Q}\nTexts:\n{D}\nRewritten question:\n"

def apply_fbk_inst_prompt(Q, D, prefix="Report:\n", *kwargs):
    p = template_report.replace('{prompt_report}', prompt_report)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p
