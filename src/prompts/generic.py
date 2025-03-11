### context template
doc_prompt_template = "[{ID}]{T}{P}\n"
def apply_docs_prompt(doc_items, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        title = doc_item.get('title', '')
        if title == '' or title is None:
            p_doc = p_doc.replace("{T}", "")
        else:
            p_doc = p_doc.replace("{T}", f" (Title: {title}) ")
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

### prompts for feedback

#### IR
def apply_fbk_inst_prompt(Q, D, prefix=None, R=None):
    prompt = \
        "Write a passage that answers the given query. {prefix} Use the provided search results to draft the answer " + \
        "(some of the search results might be irrelevant). Cite the search results if they are relevant. " + \
        "Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end."

    if prefix is not None:
        prompt = prompt.replace("{prefix}", prefix)
    else:
        prompt = prompt.replace("{prefix} ", "")

    if R is None:
        template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nPassage:\n"
        p = template.replace('{prompt}', prompt)
    else:
        template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nDraft: {R}\nNew Query:\n"
        p = template.replace('{prompt}', prompt).replace('{R}', R)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p

#### Follow-up query
def apply_followup_inst_prompt(Q, D, prefix=None):
    prompt = \
        "Answer the following question. The question requires mulitple documents to answer. " + \
        "Identify the missing information in the search results and write a follow-up query for searching that missing information. "
    template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nFollow-up Query:"
    p = template.replace('{prompt}', prompt)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p

#### Ambiguous query
def apply_asqa_inst_prompt(Q, D, prefix=None):
    prompt = \
        "Answer the following question. The question may be ambiguous and have multiple correct answers, " +\
        "and in that case, you have to provide a long-form answer including all correct answers. " + \
        "Cite the search results if they can support the answer."
    template = "{prompt}\n\nQuestion: {Q}\nSearch results:\n{D}\nAnswer:"
    p = template.replace('{prompt}', prompt)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p

#### Report generatio
def apply_report_inst_prompt(Q, D=None, R=None, prefix=None):
    prompt_0 = \
        "Write a passage that answers the given query. Use the provided search results to draft the answer " + \
        "(some of the search results might be irrelevant). " + \
        "Write the passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end."
    prompt_1 = \
        "Given a query and the draft. Refine the draft if the provided search results can fix incorrect information in it. " + \
        "(some of the search results might be irrelevant). " + \
        "Rewrite a new passage within 100 words. Add the `<p>` and `</p>` tags at the beginning and the end."
    if R is None:
        template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nPassage: <p>"
        p = template.replace('{prompt}', prompt_0)
    else:
        template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nDraft: {R}\nPassage: <p>"
        p = template.replace('{prompt}', prompt_1).replace('{R}', R)
    p = p.replace("{Q}", Q).replace("{D}", D)
    return p


### prompts for thinking models
# prompt_report_gen = "Write a passage that answers the given query: {Q}. Use your remembered documents and identify the relevant information to draft the passage. Write the passage within 100 words."
# template_report_gen = "{prompt_report}\n\n<think>\nOkay, so I need to first recall a few relevant documents that are related to the query. Let me list these docuemnts first.\n{D}\nNow, I know how to write the passage to answer the query.</think>\n\n"

### prompts for rewrite
# prompt_report_gen = "Based on the given query and the given search results (some of them might be irrelevant), generate 10 sub-queries to expand searching scope. Add the `<p>` and `</p>` tags at the beginning and the end of 10 sub-queries."
# template_report_gen = "{prompt_report}\n\nQuery: {Q}\nSearch results:\n{D}\nSub-queries:\n<p>"
#
# def apply_fbk_inst_prompt(Q, D, prefix="Report:\n"):
#     p = template_report_gen.replace('{prompt_report}', prompt_report_gen)
#     p = p.replace("{Q}", Q).replace("{D}", D)
#     return p

### prompts for feedback (old)
# prompt_report = "Write an accurate, engaging, and concise report for the given topic. Use only the provided search results (some of which might be irrelevant) and cite them properly. Always cite for any factual claim. Cite at least one document and at most three documents in each sentence."
# template_report = "{prompt_report}\n\nTopic: {Q}\n\nSearch results:\n{D}\nReport:\n:"
# prompt_report = "Elaborate the information need of the question in detail. Find the useful information in the given contexts (some of which might be irrelevant, please ignore). Write the explanation witin 50 words."
# template_report = "{prompt_report}\n\nQuestion: {Q}\n\nContexts:\n{D}\nExplanation:\n"
# prompt_report = "Rewrite the question with more comprehensive contexts, making the question easier to understand. Some useful preliminary knowledge could be found in the given texts (but some of which might be irrelevant)."
# template_report = "{prompt_report}\n\nQuestion: {Q}\nTexts:\n{D}\nRewritten question:\n"

### prompts for response
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

