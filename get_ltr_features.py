from utils_rank import *
import argparse
from bert_scores import get_bert_scores


def rewrite_features_bert(scores_dict, current_features_file, new_features_file=None):
    """
    :param scores_dict: dict of {docno: score}
    :param current_features_file: path to current features file
    :param new_features_file: path to new features file
    :return: path to new features file
    """
    if new_features_file is None:
        new_features_file = current_features_file + "_with_bert"

    # normalize scores
    min_score, max_score = min(scores_dict.values()), max(scores_dict.values())
    if max_score == min_score:
        normalized_scores = {key: 0 for key in scores_dict}
    else:
        normalized_scores = {}
        for key in scores_dict:
            normalized_scores[key] = (scores_dict[key] - min_score) / (max_score - min_score)

    # get current features
    with open(current_features_file, 'r') as f:
        old_features_lines = f.readlines()

    # add bert scores to features file
    with open(new_features_file, 'w') as f:
        for line in old_features_lines:
            prev_features, suffix = line.split("#")
            docno = suffix.strip()
            new_line = prev_features + "26:" + str(normalized_scores[docno]) + " #" + suffix
            f.write(new_line)

    # update featureID file
    with open('featureID', 'w') as f:
        f.write('BertScore:26\n')

    return new_features_file


def build_index(documents_trectext: str, index_path: str):
    """
    Parse the trectext file given, and create an index.
    :param documents_trectext: the trectext file
    :return: the path to the index
    """
    _ = subprocess.Popen("rm -r " + index_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         shell=True).communicate()
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    # parameters for IndriBuildIndex
    build_index_params = {
        '-corpus.path': documents_trectext,
        '-corpus.class': 'trectext',
        '-memory': '1G',
        '-index': index_path,
        '-stemmer.name': 'krovetz'
    }

    # Build the index

    command_params = ' '.join([f"{key}={value}" for key, value in build_index_params.items()])

    out, _ = subprocess.Popen(indri_build_index + " " + command_params, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              shell=True).communicate()

    return out


def run_LTRFeatures(queries_file_path: str, index_path: str, working_set_file: str, features_dir: str = 'features_dir'):
    """
    Run LTRFeatures on the index and queries file.
    :param queries_file_path: the queries file
    :param index_path: the path to the index
    :param working_set_file: the working set file
    :param features_dir: the directory to save the features
    :return: the output of the command
    """
    _ = subprocess.Popen("rm -r " + features_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         shell=True).communicate()
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # parameters for LTRFeatures
    LTRFeatures_params = {
        '-stream': 'doc',
        '-index': index_path,
        '-repository': index_path,
        '-useWorkingSet': 'true',
        '-workingSetFile': working_set_file,
        '-workingSetFormat': 'trec',
    }

    # Run LTRFeatures
    LTRFeatures_module_path = 'scripts/LTRFeatures'
    command_params = ' '.join([f"{key}={value}" for key, value in LTRFeatures_params.items()])

    out, err = subprocess.Popen(LTRFeatures_module_path + ' ' + queries_file_path + ' ' + command_params,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True).communicate()

    _ = iter(subprocess.Popen(f'mv doc*_* {features_dir}', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              shell=True).stdout.readline, b'')
    return out


def run_generate_pl(features_dir: str, working_set_file: str):
    """
    Run generate.pl
    :param features_dir: the directory of the features
    :param working_set_file: the working set file
    :return: the output of the command
    """

    # Run generate.pl
    generate_pl_path = 'scripts/generate.pl'
    out, _ = subprocess.Popen('perl ' + generate_pl_path + ' ' + features_dir + ' ' + working_set_file,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True).communicate()

    return out


def write_working_set_file(queries_to_docnos: dict, working_set_file_path: str):
    with open(working_set_file_path, 'w') as f:
        for query, docnos in queries_to_docnos.items():
            for i, docno in enumerate(docnos):
                f.write(query.zfill(3) + ' Q0 ' + docno + ' ' + str(i + 1) + ' -' + str(i + 1) + ' indri\n')


def write_queries_xml(queries_xml_file_path: str, queries_text_file_path: str):
    """
    Write the all_queries to an xml file.
    :param queries_xml_file_path: the path to the xml file
    :param queries_text_file_path: the path to the text file
    """
    with open(queries_text_file_path, 'r') as f:
        all_queries = f.readlines()

    all_queries = [query.split(":") for query in all_queries]
    all_queries = {query[0]: query[1].strip() for query in all_queries}

    with open(queries_xml_file_path, 'w') as f:
        f.write("<parameters>\n")
        # for q_id in sorted(list(queries)):
        for q_id in sorted(list(all_queries.keys())):
            f.write("<query>\n")
            f.write("<number>" + str(q_id).zfill(3) + "</number>\n")
            f.write("<text>#combine( " + all_queries[q_id] + " )</text>\n")
            f.write("</query>\n")
        f.write("</parameters>\n")


def write_qrels(qrels_file_path: str, features_file: str):
    new_file = features_file + "_train"
    rel_df = pd.read_csv(qrels_file_path, sep=' ', header=None, names=['query_id', 'Q0', 'docno', 'rel'])
    rel_df['docno'] = rel_df['docno'].apply(lambda x: 'ROUND' + x.strip('EPOCH') if x.startswith('EPOCH') else x)

    with open(features_file, 'r') as f:
        features = f.readlines()

    with open(new_file, 'w') as f:
        for feature in features:
            line = feature[2:]
            docno = line.split("#")[1].strip()
            query_id = line.split("qid:")[1].split(' ')[0]

            rel = rel_df[(rel_df['query_id'].astype(str).str.zfill(len(str(query_id))) == str(query_id)) &
                         (rel_df['docno'] == docno)]['rel'].values[0]

            new_line = str(
                int(rel)) + ' ' + line
            f.write(new_line)

    return new_file


def get_queries_dict(queries_text_file_path):
    with open(queries_text_file_path, 'r') as f:
        all_queries = f.readlines()

    all_queries = [query.split(":") for query in all_queries]
    all_queries = {query[0]: query[1].strip() for query in all_queries}

    return all_queries


def get_ltr_features(docs_trectext_file_path: str = 'merged/documents.trectext',
                     qrels_file_path: str = 'merged/documents.rel',
                     features_file: str = 'features_output',
                     print_commands: bool = False,
                     mode: str = 'train',
                     queries_text_file_path: str = 'all_queries.txt',
                     working_set_file_path: str = 'working_set.txt'):
    """
    Get the LTR features.
    """
    print_commands and print("Load data...")
    docs_dict = read_trectext(docs_trectext_file_path)  # dict of docno:doc_text
    queries_to_docnos = get_query_to_docno(docs_dict)  # dict of query:docnos

    print_commands and print("Write queries xml file...")
    queries_xml_file_path = 'queries.xml'  # 'data/queries.xml'
    write_queries_xml(queries_xml_file_path, queries_text_file_path)
    queries_dict = get_queries_dict(queries_text_file_path)

    print_commands and print("Write working set file...")
    write_working_set_file(queries_to_docnos, working_set_file_path)

    # create index
    print_commands and print("Create index...")
    index_path = 'foo_index'
    command_out = build_index(docs_trectext_file_path, index_path)
    print_commands and print_command_out(command_out)

    # create features file
    print_commands and print("Create features using LTRFeatures module...")
    features_dir = 'features_dir'
    command_out = run_LTRFeatures(queries_xml_file_path, index_path, working_set_file_path, features_dir)
    print_commands and print_command_out(command_out)

    print_commands and print("Generate pl...")
    command_out = run_generate_pl(features_dir, working_set_file_path)
    print_commands and print_command_out(command_out)

    # add bert ranker scores to features
    print_commands and print("Add bert scores...")
    bert_scores = get_bert_scores(queries_to_docnos, docs_dict, queries_dict)
    features_file = rewrite_features_bert(bert_scores, current_features_file='./features',
                                          new_features_file=features_file)

    if mode == 'test':
        print_commands and print("Done!")
        print_commands and print(f"Path to Test File: {features_file}")
        return features_file

    train_file = write_qrels(qrels_file_path=qrels_file_path, features_file=features_file)

    print_commands and print(f"Path to Training File: {train_file}")
    print_commands and print("Done!")
    return train_file


if __name__ == "__main__":

    arguments = argparse.ArgumentParser()
    arguments.add_argument('--docs_trectext', type=str, default='merged/documents.trectext',
                           help='The path to the trectext file')
    arguments.add_argument('--qrels_file', type=str, default='merged/documents.rel',
                            help='The path to the qrels file')
    arguments.add_argument('--queries_text_file_path', type=str, default='data/all_queries.txt',
                            help='The path to the queries text file')
    arguments.add_argument('--working_set_file_path', type=str, default='data/working_set.txt',
                            help='The path to the working set file')
    arguments.add_argument('--features_file', type=str, default='features_with_bert',
                            help='The path to the features file without qrels')
    arguments.add_argument('--print_commands', type=bool, default=False,
                            help='Print the commands')
    arguments.add_argument('--mode', type=str, default='train', help='train or test', choices=['train', 'test'])
    args = arguments.parse_args()

    docs_trectext_file_path = args.docs_trectext
    qrels_file_path = args.qrels_file
    features_file = args.features_file
    print_commands = args.print_commands
    queries_text_file_path = args.queries_text_file_path
    working_set_file_path = args.working_set_file_path

    get_ltr_features(docs_trectext_file_path=docs_trectext_file_path,
                        qrels_file_path=qrels_file_path,
                        features_file=features_file,
                        print_commands=print_commands,
                        mode=args.mode,
                        queries_text_file_path=queries_text_file_path,
                        working_set_file_path=working_set_file_path)


