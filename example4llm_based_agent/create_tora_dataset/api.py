from utils import extract_program
from keys import openai_key as OPENAI_KEY


def llm_api(engine, prompt, max_tokens, temperature, n, top_p, stop):
    pass


def api_with_func_call(engine, prompt, max_tokens, temperature, n, top_p, executor, max_func_call=4, verbose=False):
    if n > 1:
        assert temperature > 0

    if verbose:
        print("\n======= API with function call (START) =======")

    next_batch_queries = [""] * n
    end_queries = []
    for i in range(max_func_call):
        batch_outputs = []
        batch_queries = next_batch_queries
        if len(batch_queries) == 0:
            break
        # get all outputs
        # support batch inference when n > 1
        if i == 0:
            results = llm_api(
                engine=engine,
                prompt=prompt + batch_queries[0],
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                top_p=top_p,
                stop=["```output\n", "---"],
            )
            batch_outputs.extend(results)
        else:
            for k, query in enumerate(batch_queries):
                print("Call {} / {}".format(k+1, len(batch_queries)))
                results = llm_api(
                    engine=engine,
                    prompt=prompt + query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                    top_p=top_p,
                    stop=["```output\n", "---"],
                )
                batch_outputs.append(results[0])

        # process all outputs
        next_batch_queries = []
        for query, output in zip(batch_queries, batch_outputs):
            output = output.rstrip()
            query += output
            if verbose:
                print("\n", "-" * 20)
                print(output, end="")
            if "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                prediction, report = executor.apply(program)
                exec_result = prediction if prediction else report
                exec_result = f"\n```output\n{exec_result.strip()}\n```\n"
                query += exec_result
                if verbose:
                    print(exec_result, end="")
                # not end
                if i == max_func_call - 1:
                    query += "\nReach max function call limit."
                next_batch_queries.append(query)
            else:
                end_queries.append(query)

    end_queries.extend(next_batch_queries)
    return end_queries
