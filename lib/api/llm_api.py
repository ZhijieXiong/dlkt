from openai import OpenAI
import tiktoken


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """
    Return the number of tokens used by a list of messages.

    source: https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def prompt_chat(openai_key, model_name, prompt, **kwargs):
    client = OpenAI(api_key=openai_key)
    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )

    return completion.choices[0].message


def zero_or_few_shot(openai_key, model_name, profile_message, examples, query, messages_type, **kwargs):
    """
    source: https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models

    :param openai_key: str
    :param model_name: str
    :param profile_message: 描述LLM，如扮演的角色
    :param examples: few shot的例子，格式为List[(query: str, answer: str)]
    :param query: str
    :param messages_type: str, "real" or "fake"
    :return: completion.choices[0].message
    """
    client = OpenAI(api_key=openai_key)
    messages = []
    if type(profile_message) is str or profile_message != "":
        messages.append({
            "role": "system",
            "content": profile_message
        })
    if (type(examples) is list) and (len(examples) > 0):
        for example_query, example_answer in examples:
            if messages_type == "real":
                message_query = {
                    "role": "system",
                    "name": "example_user",
                    "content": example_query
                }
                message_answer = {
                    "role": "system",
                    "name": "example_assistant",
                    "content": example_answer
                }
            elif messages_type == "fake":
                message_query = {
                    "role": "user",
                    "content": example_query
                }
                message_answer = {
                    "role": "assistant",
                    "content": example_answer
                }
            else:
                message_query = {
                    "role": "system",
                    "name": "example_user",
                    "content": example_query
                }
                message_answer = {
                    "role": "system",
                    "name": "example_assistant",
                    "content": example_answer
                }
            messages.append(message_query)
            messages.append(message_answer)
    messages.append({
        "role": "user",
        "content": query
    })

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )

    return completion.choices[0].message


