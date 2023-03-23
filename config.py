import openai

data_exist = {"comp": True, "improvements": True, "tops": True}
display_graphs = False

openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

prompt_ = "Please write a short document of no more than 150 words on the topic of {topic}. " \
          "Your document should cover the following subtopics: {subtopics[0]}," \
          "{subtopics[1]}, {subtopics[2]} Your writing should be informative and engaging, and should provide " \
          "the reader with a clear understanding of the topic and its related subtopics."

prompt = prompt_
model = "text-davinci-003"
temperature = 0.7
top_p = 1.0
max_tokens = 900
frequency_penalty = 0.0
presence_penalty = 1
