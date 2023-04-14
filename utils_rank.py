import os
import subprocess
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

indri_build_index = './scripts/indri-5.6/buildindex/IndriBuildIndex'
ranklib = 'scripts/RankLib.jar'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_command_out(command_out: bytes, indent: int = 1):
    start = "\t" * indent
    new_line = "\n" + start
    print(start + command_out.decode("utf-8").replace("\n", new_line).replace("\r", new_line))


def read_trectext(file="documents.trectext"):
    """
    Read the documents from the trectext file.

    Format:
            <DOC>
            <DOCNO>ROUND-<round_number>-<query_id>-<author_id></DOCNO>
            <TEXT>text</TEXT>
            </DOC>
    Example:
            <DOC>
            <DOCNO>ROUND-00-195-00</DOCNO>
            <TEXT>
            And it's a good thing. Because living Out Here, there's no end to the ways we put ours to use.
            That's why Tractor Supply carries a full line of powerful, performance-oriented pressure washers in a range of different sizes, configurations and horsepowers. As well as all the accessories and connectors you need to hit the ground spraying.
            We carry cold water residential pressure washers in sizes that can handle any project on your list – from 1700 PSI at 1.8 GPM all the way up to 4200 PSI at 4.0 GPM. And commercial hot water skid mount units that can handle even the toughest jobs. All in addition to replacement and extension hoses, quick disconnect adapters, spray guns and nozzles, pressure regulators and crankcase oil. Not to mention a wide range of spray tips and power wash concentrates.
            </TEXT>
            </DOC>

    :param file: name of the trectext file
    :return: dictionary with the document number as the key and the document text as the value
    """
    with open(file, "r", encoding='utf-8', errors='ignore') as f:
        utf8_str = f.read()

    documents = {}
    for doc_block in utf8_str.split('<DOC>')[1:]:
        docno = doc_block.split('<DOCNO>')[1].split('</DOCNO>')[0]
        text = doc_block.split('<TEXT>')[1].split('</TEXT>')[0].strip()
        documents[docno] = text
    # print(f"Read {len(documents)} documents from {file}")
    return documents


def get_query_to_docno(docs_dict):
    """
    Returns a dictionary mapping query_id to a list of docnos for that query.
    """
    query_to_docno = {}
    for docno, doc in docs_dict.items():
        _, query_id, _ = decode_docno(docno)
        if query_id not in query_to_docno:
            query_to_docno[query_id] = []
        query_to_docno[query_id].append(docno)
    return query_to_docno


def decode_docno(docno: str):
    """
    Decode the document ID to get the round number, query ID and author ID.
    Format: ROUND-<round_number>-<query_id>-<author_id>
    :param docno: document ID
    :return: round number, query ID and author ID
    """
    round_number, query_id, author_id = docno.split('-')[1:]
    return round_number, query_id, author_id


def write_trectext(data: pd.DataFrame, dir_path: str = './', file_name: str = "documents.trectext"):
    with open(dir_path + '/' + file_name, 'w', encoding='utf-8', errors='ignore') as f:
        for _, row in data.iterrows():
            docno = row['merged_docno']
            text = row['text']
            f.write('<DOC>\n')
            f.write('<DOCNO>' + docno + '</DOCNO>\n')
            f.write('<TEXT>\n' + f'{text}' + '\n</TEXT>\n')
            f.write('</DOC>\n')
