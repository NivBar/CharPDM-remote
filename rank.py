from typing import List, Dict
from get_ltr_features import get_ltr_features
from utils_rank import *
import os




def rank(features_file, model_path):
    scores_file = features_file + '.score'
    rank_params = {
        '-load': model_path,
        '-rank': features_file,
        '-indri': scores_file,
    }

    eval_command = 'java -jar ' + ranklib + ' ' + ' '.join(
        [f"{key} {value}" for key, value in rank_params.items()])

    out, _ = subprocess.Popen(eval_command, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              shell=True).communicate()

    scores = pd.read_csv(scores_file, sep=' ', header=None,
                         names=['query_id', 'Q0', 'docno', 'rank', 'score', 'run_id'])
    return scores


def rank_documents(query: str, documents: Dict[str, str], model_path: str = "scripts/LambdaMART_model") -> Dict[str, int]:
    """
    Given a query and a list of documents, rank the documents using the model.
    :param query: The query to rank the documents for.
    :param documents: A dictionary mapping document ids to their text.
    :param model_path: The path to the model to use for ranking.
    :return: A dictionary mapping document ids to their rank.
    """
    temp_dir = 'temp_dir'
    _ = subprocess.Popen("rm -r " + temp_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         shell=True).communicate()
    os.makedirs(temp_dir)

    internal_docnos = {f'ROUND-01-001-{docno}@internal': docno for docno in documents.keys()}
    internal_docnos.update({v: k for k, v in internal_docnos.items()})

    documents_df = pd.DataFrame.from_dict(documents, orient='index', columns=['text'])
    documents_df.index.name = 'docno'
    documents_df['merged_docno'] = documents_df.index.map(internal_docnos)

    write_trectext(documents_df, temp_dir)
    docs_trectext_file_path = temp_dir + '/documents.trectext'

    queries_text_file_path = temp_dir + '/queries.txt'
    with open(queries_text_file_path, 'w') as f:
        f.write(f'001:{query}')

    output_file = get_ltr_features(docs_trectext_file_path=docs_trectext_file_path,
                                   queries_text_file_path=queries_text_file_path,
                                   working_set_file_path=temp_dir + '/working_set.txt',
                                   features_file=temp_dir + '/features_output',
                                   print_commands=False,
                                   mode='test')

    scores_df = rank(output_file, model_path)
    return {internal_docnos[docno]: r for docno, r in scores_df.set_index('docno')['rank'].to_dict().items()}


# if __name__ == '__main__':
#     query = "a sample query"
#     documents = {
#         'doc1': 'this is a sample document that is very long and has a lot of words in it',
#         'doc3': 'who wants pizza',
#         'doc2': 'another sample document that is also very long and has a lot of words in it',
#         'doc4': 'a sample query',
#     }
#
#     rankings = rank_documents(query, documents)
#     print(rankings)
