import json
import os
import competition_chatgpt as cgpt
import utils
import config
from rank import rank_documents
from pprint import pprint

id_ = 193
epoch = 3
bot_type = "tops"
markov = False
bot_name = config.get_names_dict(markov)[bot_type]
competitors = utils.get_id_text_pairs(id_=id_, epoch=epoch)
query = config.query_index[id_]

dir_path = f"./epoch_data/{bot_name}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

print(f"Handling {id_}: {query}, Epoch: {epoch}, bot_name: {bot_name} (type: {bot_type} & markov: {markov})")
print("Creating prompt...")
messages = cgpt.get_messages(id_, epoch=epoch, bot_type=bot_type, markov=markov)
pprint(messages)
print("Generating text...")
res = cgpt.get_comp_text(messages)['choices'][0]['message']['content']
print(f"Ranking...")
competitors.update({bot_name: res})
competitors = {k.split("-")[-1]: v for k, v in competitors.items()}
rankings = rank_documents(query=query, documents=competitors)
print(rankings)

with open(f"{dir_path}/id_{id_}_ep_{epoch}_{bot_name}_texts.json", "w") as outfile:
    outfile.write(json.dumps(competitors, indent=4))

with open(f"{dir_path}/id_{id_}_ep_{epoch}_{bot_name}_rankings.json", "w") as outfile:
    outfile.write(json.dumps(rankings, indent=4))

x = 1
# format to rank - {'id1': 'text1', 'id2': 'text2'}
