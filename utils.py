import openai
import pandas as pd
import config


def get_initial_doc(topic, subtopics):
    response = False
    prompt_ = fr"Please write a short document of no more than 150 words on the topic of {topic}. " \
              f"Your document should cover the following subtopics: {subtopics[0]}," \
              f"{subtopics[1]}, {subtopics[2]} Your writing should be informative and engaging, and should provide " \
              f"the reader with a clear understanding of the topic and its related subtopics."

    while not response:
        try:
            response = openai.Completion.create(
                model=config.model,
                prompt=config.prompt_,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            # print("success")
            break
        except Exception as e:
            print(e)
            continue
    return response


def data_set_creation():
    with open("..\content_modification_dataset\documents.trectext", "r", encoding="utf8") as f:
        xml = f.read().replace("&", "&amp;")
    xml = fr"<root>{xml}</root>"
    doc_df = pd.read_xml(xml).astype(str)

    qrel_df = \
        pd.read_csv("..\content_modification_dataset\documents.quality", header=None, delimiter=r"\s+").astype(
            str).rename(
            {2: "DOCNO", 3: "QREL"}, axis=1).replace(
            {'EPOCH': 'ROUND'}, regex=True)[["DOCNO", "QREL"]]

    ksrels_df = \
        pd.read_csv("..\content_modification_dataset\documents.relevance", header=None, delimiter=r"\s+").astype(
            str).rename({2: "DOCNO", 3: "KSREL"}, axis=1)[
            ["DOCNO", "KSREL"]]

    query_df = pd.read_csv("..\content_modification_dataset\queries.txt", header=None, delimiter=r":").astype(
        str).rename(
        {0: "query_id", 1: "query"},
        axis=1)

    pos_df = \
        pd.read_csv("..\content_modification_dataset\documents.positions", header=None, delimiter=r"\s+").astype(
            str).rename(
            {2: "DOCNO", 3: "POS"}, axis=1)[["DOCNO", "POS"]]

    merge_df = doc_df.merge(qrel_df, on="DOCNO").merge(ksrels_df, on="DOCNO").merge(pos_df, on="DOCNO")
    merge_df[["round_number", "query_id", "author_id"]] = merge_df["DOCNO"].apply(
        lambda x: pd.Series(str(x).split("-")[1:]))
    merge_df = merge_df.merge(query_df, on="query_id")[
        ['DOCNO', 'round_number', 'query_id', 'author_id', 'query', 'TEXT', 'QREL', 'KSREL', 'POS']]

    return merge_df
