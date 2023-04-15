import openai
import config
import tiktoken
import json

encoder = tiktoken.encoding_for_model(config.model)


def rank_suff(loc):
    if loc == 1:
        return ("st")
    elif loc == 2:
        return ("nd")
    elif loc == 3:
        return ("rd")
    else:
        return ("th")


def get_messages(idx: int, epoch=1, bot_type="all", markov=False):
    """
    craft prompt messages list for the bot according to its type
    :param idx: topic index
    :param epoch: current epoch number
    :param bot_type: "all" gives the bot documents and rankings along all epochs,
                     "tops" gives the bot only the top document along all epochs,
                     "self" gives the bot only the ranking it is in, in every epoch along all epochs
    :param markov: looking only on the last round
    :return: list of messages comprising the prompt
    """
    assert type(idx) == int
    assert type(epoch) == int
    assert bot_type in ["all", "tops", "self"]
    assert type(markov) == bool

    # topic_info = config.topic_codex[idx]

    # csv option:
    topic_info = {}
    rel = config.comp_data[config.comp_data.query_id == idx].head(1)
    topic_info['queries'] = [rel["query"][0]]
    topic_info['doc'] = rel['TEXT'][0]
    bot_name = config.get_names_dict(markov)[bot_type]

    messages = [
        {"role": "system", "content": f"You are a contestant in an information retrieval SEO competition."},
        {"role": "system",
         "content": fr"The competition involves {len(topic_info['queries'])} queries: " + ",".join(
             topic_info['queries'])},
        {"role": "system",
         "content": "The goal is to have your document be ranked 1st (first) and win in the ranking done by a black "
                    "box ranker in every single epoch."},
        {"role": "system", "content": fr"All contestants got an initial reference text: '{topic_info['doc']}'."},
        {"role": "user",
         "content": "Generate a single text that addresses the information need for all queries."},
    ]
    if epoch != 1:
        min_ = epoch - 1 if markov else 1
        for i in range(min_, epoch):
            rankings = json.load(open(f'./epoch_data/{bot_name}/id_{idx}_ep_{i}_{bot_name}_rankings.json'))
            top_user = [x for x in rankings.keys() if rankings[x] == 1][0]
            curr_rank = rankings.pop(bot_name)
            texts = json.load(open(f'./epoch_data/{bot_name}/id_{idx}_ep_{i}_{bot_name}_texts.json'))
            top_text = texts[top_user]
            curr_text = texts.pop(bot_name)

            messages.append({"role": "assistant", "content": f"{curr_text}"})
            messages.append(
                {"role": "system", "content": f"You were ranked {curr_rank}{rank_suff(curr_rank)} in this epoch"})

            if bot_type == "all":
                txt_rnk = str(
                    {f"ranked {obj[0]}": f"{obj[1]}\n\n" for obj in zip(rankings.values(), texts.values())}).replace(
                    "\'", "").replace("\"", "")
                messages.append(
                    {"role": "system",
                     "content": f"The ranked documents of your opponents in this epoch are as follows:\n {txt_rnk}"})
                messages.append({"role": "user",
                                 "content": "Generate a single text that addresses the information need for "
                                            "all queries."})

            elif bot_type == "tops":
                messages.append(
                    {"role": "system", "content": f"The document ranked 1 in this epoch was: {top_text}"})
                messages.append({"role": "user",
                                 "content": "Generate a single text that addresses the information need for "
                                            "all queries."})

            elif bot_type == "self":
                messages.append({"role": "user",
                                 "content": "Generate a single text that addresses the information need for "
                                            "all queries."})
    return messages


def get_comp_text(messages):
    max_tokens = config.max_tokens
    response = False
    prompt_tokens = len(encoder.encode("".join([line['content'] for line in messages]))) + 200
    while prompt_tokens + max_tokens > 4096:
        max_tokens -= 50
        print("Changed max tokens for response to:", max_tokens)

    while not response:
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=max_tokens,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
            )
            # print("success")
            word_no = len(response['choices'][0]['message']['content'].split())
            if word_no > 150:
                max_tokens -= 50
                response = False
                print(f"word no was: {word_no}, dropping max tokens to: {max_tokens}.")
                continue
            break
        except Exception as e:
            print(e)
            continue
    return response

# if __name__ == '__main__':
#     messages = get_messages(193, epoch=4, bot_type="all", markov=False)
#     res = get_comp_text(messages)['choices'][0]['message']['content']
#     x = 1
