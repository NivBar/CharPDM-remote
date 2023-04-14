from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and tokenizer
tokenizer_name = 'bert-large-uncased'
model_name = "castorini/monobert-large-msmarco"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()


def get_bert_scores(queries_to_docnos, docs_dict, queries_dict):
    scores_dict = {}
    # for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
    for qid in queries_to_docnos:
        for docno in queries_to_docnos[qid]:
            query_text = queries_dict[qid]
            doc_text = docs_dict[docno]
            ret = tokenizer.encode_plus(query_text,
                                        doc_text,
                                        max_length=512,
                                        truncation=True,
                                        return_token_type_ids=True,
                                        return_tensors='pt')

            with torch.cuda.amp.autocast(enabled=False):
                input_ids = ret['input_ids'].to(device)
                tt_ids = ret['token_type_ids'].to(device)
                output, = model(input_ids, token_type_ids=tt_ids, return_dict=False)
                if output.size(1) > 1:
                    score = torch.nn.functional.log_softmax(output, 1)[0, -1].item()
                else:
                    score = output.item()

            scores_dict[docno] = score

    return scores_dict
