from utils import extract_program
import os
import json
import inspect
from openai import OpenAI
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "../../keys.json"), "r") as f:
    KEYS = json.load(f)
OPENAI_KEY = KEYS["openai_key_from_lwd2hzhp"]


def llm_api(model_name, prompt, max_tokens, temperature, n, top_p, stop):
    assert model_name in ["gpt-3.5-turbo", "gpt-4"], f"{model_name} is not support now"
    client = OpenAI(api_key=OPENAI_KEY)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=2,
        top_p=top_p,
        stop=stop
    )

    return list(map(lambda c: c.message.content, completion.choices))


def api_with_func_call(model_name, prompt, max_tokens, temperature, n, top_p, executor, max_func_call=4):
    assert n > 0
    if n > 1:
        # n > 1表示多次生成，需要temperature参数（0～2，值越大越输出越随机，建议值0.8）或者top_p参数
        assert (0 < temperature < 2) and (top_p == 1), f"open ai: We generally recommend altering top_p or temperature but not both."
        assert (top_p > 1) and (temperature == 1), f"open ai: We generally recommend altering top_p or temperature but not both."

    next_batch_queries = [""] * n
    # end_queries的element是chatgpt的输出（不包含prompt），它的数量和参数n一样
    end_queries = []
    for i in range(max_func_call):
        batch_outputs = []
        batch_queries = next_batch_queries
        if len(batch_queries) == 0:
            # 如果已经生成了需要的内容（"boxed" in output），则不继续调用api
            break

        if i == 0:
            results = llm_api(
                model_name=model_name,
                prompt=prompt + batch_queries[0],
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                top_p=top_p,
                stop=["```output\n", "---"],
            )
            batch_outputs.extend(results)
        else:
            # 将前面的输出加到这次的prompt中
            for k, query in enumerate(batch_queries):
                results = llm_api(
                    model_name=model_name,
                    prompt=prompt + query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                    top_p=top_p,
                    stop=["```output\n", "---"],
                )
                # 这里从第二次开始，只要top 1的输出（在第一次调用时，如果n>1，会把所有结果保留），减少调用api
                batch_outputs.append(results[0])

        # process all outputs
        next_batch_queries = []
        for query, output in zip(batch_queries, batch_outputs):
            output = output.rstrip()
            query += output
            if "boxed" not in output and output.endswith("```"):
                # 如果输出没有包含最终结果，并且是一段程序，则执行程序并提取结果，将输出和结果拼接起来，
                # 作为下一次chatgpt的输入，让它来生成最终的结果
                program = extract_program(query)
                prediction, report = executor.apply(program)
                exec_result = prediction if prediction else report
                exec_result = f"\n```output\n{exec_result.strip()}\n```\n"
                query += exec_result
                if i == max_func_call - 1:
                    query += "\nReach max function call limit."
                next_batch_queries.append(query)
            else:
                end_queries.append(query)
    end_queries.extend(next_batch_queries)

    return end_queries
