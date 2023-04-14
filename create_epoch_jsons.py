import json
import os
import competition_chatgpt as cgpt
import utils
import config

id_ = 193
epoch = 4
bot_type = "all"
markov = False
bot_name = config.get_names_dict(markov)[bot_type]
competitors = utils.get_id_text_pairs(id_=id_, epoch=epoch)

if not os.path.exists(f"./epoch_data/{bot_name}"):
    os.makedirs(f"./epoch_data/{bot_name}")

messages = cgpt.get_messages(193, epoch=4, bot_type="all", markov=False)
res = cgpt.get_comp_text(messages)['choices'][0]['message']['content']

# format to rank - {'id1': 'text1', 'id2': 'text2'}

