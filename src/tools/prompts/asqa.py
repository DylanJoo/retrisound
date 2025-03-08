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

#### Query-decompose
def apply_break_inst_prompt(Q, D, prefix=None):
    prompt = \
        "Based on the search results, break down the query into multiple sub-queries. " + \
        "Each sub-queries needs to cover the important points about the query, "+ \
        "allowing to make the following report based on these points. " + \
        "List more sub-queries if the original query is vague or ambiguous."
    template = "{prompt}\n\nQuery: {Q}\nSearch results:\n{D}\nSub-queries:"
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
