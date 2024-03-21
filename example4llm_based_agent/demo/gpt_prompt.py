from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
    # print("chatgpt message",chatgpt_messages)
    url = "https://www.jiujiuai.life/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
        "Authorization": "sk-XHx1xQu7iM5kmgYVFd08Ce8f3e5a4dEa8c2c00Bf62225244"
    }

    data = {
        "model": model,
        "messages": chatgpt_messages,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=data, headers=headers)
    # response = requests.post("https://www.jiujiuai.life/v1/chat/completions", json=chatgpt_messages)
    # response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens)
    # response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens)
    response_data = response.json()
    # import pdb
    # pdb.set_trace()
    reply = response_data['choices'][0]['message']['content']
    total_tokens = response_data['usage']['total_tokens']
    return reply, total_tokens


prompt = "In this task, you are given an input sentence. Your job is to output all the object noun phrases. \
            Input: there is a brown table sitting in the corner. it has a couch to its right. \
            Output: ['a brown table', 'a couch'] \
            Input: in a center of the room are two identical coffee tables , this may be the coffee table on left this may be the coffee table on the right .  it is and the small coffee table that is furthest away from the small grey picture on the wall. \
            Output: ['two identical coffee tables', 'the coffee table', 'the coffee table', 'the small coffee table', 'the small grey picture', 'the wall'] \
            Input: image shows a burgundy couch near an off-white wall.  two wooden side tables are directly in front of the couch.  directly across from the pictured couch is a duplicate couch.  diagonally to the left and right of the couch are 2 sets of single chairs that are the same color and material as the couch. \
            Output: ['a burgundy couch', 'an off-white wall', 'two wooden side tables', 'the couch', 'the pictured couch', 'a duplicate couch', ,'the couch', '2 sets of single chairs', 'the couch'] \
            Input: the radiator is embedded in the wall, where a chair with a pillow in it, that is located right in front of radiator. a large picture frame is located right above the radiator, which is where a chair is at with a pillow on it. the radiator is located on the wall where a frame picture hangs above it, this radiator has slim vents located in from of radiator. \
            Output: ['the radiator', 'the wall', 'a chair', 'a pillow', 'radiator', 'a large picture frame', 'the radiator', 'a chair', 'a pillow', 'the radiator'. 'the wall', 'a frame picture', 'this radiator', 'slim vents', 'radiator'] \
            Input: this is large machine. lit is next to the black garbage bin in the corner. it is the largest machine in the room,it has a small lit up screen with some letters or numbers. \
            Output: "
message = [{"role": "user", "content": prompt}]
# 注意max_tokens不能太大
question, n_tokens = call_chatgpt(message, model="gpt-3.5-turbo", max_tokens=3000)
print(question)
