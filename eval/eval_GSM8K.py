JSON_PATH = "/home/seungyoun/llama_related/llama_code_interpreter/eval/grade-school-math/grade_school_math/data/test.jsonl"

import os, sys
import random
import json
import re
from tqdm import tqdm


def load_jsonl(file_path, num_lines_to_display=5):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            out_dict = json.loads(line)
            answer_raw = out_dict["answer"]
            answer_refined = answer_raw.split("####")[-1].strip()
            out_dict["answer_refined"] = answer_refined
            data.append(out_dict)  # dict_keys(['question', 'answer'])
    return data


few_shot_1 = [
    {
        "role": "user",
        "content": "Sandra wants to buy some sweets. She saved $10 for this purpose. Her mother gave her an additional $4, and her father twice as much as her mother. One candy costs $0.5, and one jelly bean $0.2. She wants to buy 14 candies and 20 jelly beans. How much money will she be left with after the purchase?",
    },
    {
        "role": "assistant",
        "content": "```python\n# Saved Money\nsandra_money = 10\n# mother gave her\nmother_money = 4\n# father gave twice as much as her mother\nfather_money = 2 * mother_money\n# Cost\ncandy_cost = 0.5\njelly_bean_cost = 0.2\n\n# Calculations\ntotal_money = sandra_money + mother_money + father_money\n#Desired to buy\nShe wants to buy 14 candies and 20 jelly beans so the total cost would be\ntotal_cost = 14 * candy_cost + 20 * jelly_bean_cost\n# After the purchase\nremaining_money = total_money - total_cost\n\nprint(remaining_money)\n```\n```RESULT\n11.0\n```\nAfter buying 14 candies and 20 jelly beans, Sandra will be left with $11.",
    },
]

few_shot_2 = [
    {
        "role": "user",
        "content": "Keanu bought some fish from the store for his pets. He gave his cat half as many fish as he gave to his dog. If he gave his dog 40 fish, and each fish cost him $4, calculate the total amount of money he spent at the store to buy all the fish.",
    },
    {
        "role": "assistant",
        "content": "```python\ndef calculate_total_cost(dog_fish, fish_price):\n    cat_fish = dog_fish / 2\n    total_fish = dog_fish + cat_fish\n    total_cost = total_fish * fish_price\n    return total_cost\n\ntotal_cost = calculate_total_cost(40, 4)\nprint(total_cost)\n```\n```RESULT\n240.0\n```\nKeanu spent a total of $240 to buy all the fish from the store.\n### Answer : 240",
    },
]

few_shot_3 = [
    {
        "role": "user",
        "content": "Mike is feeding the birds at the park. He counts out the seeds as he throws them. He throws 20 seeds to the birds on the left. He throws twice as much to the bigger group of birds on the right. Some more birds flutter over to join the others for lunch and Mike throws 30 more seeds for them. If Mike has 30 seeds left to feed the last of the birds, how many seeds did Mike start with?",
    },
    {
        "role": "assistant",
        "content": "Let's break down the problem:\n\n- Mike throws 20 seeds to the birds on the left.\n- He throws twice as much to the bigger group of birds on the right, which is 2 * 20 = 40 seeds.\n- Some more birds join and Mike throws 30 more seeds for them.\n- Mike has 30 seeds left to feed the last of the birds.\n\nSo, the total number of seeds Mike started with is the sum of all these seeds.\n\n```python\nseeds_left = 20\nseeds_right = 2 * seeds_left\nseeds_more_birds = 30\nseeds_last = 30\n\ntotal_seeds = seeds_left + seeds_right + seeds_more_birds + seeds_last\nprint(total_seeds)\n```\n```RESULT\n120\n```\nFrom the calculations, we can conclude that Mike started with 120 seeds to feed the birds at the park.\n### Answer : 120",
    },
]

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter
    import argparse

    parser = argparse.ArgumentParser(description="Process path for LLAMA2_FINETUNEED.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the finetuned LLAMA2 model.",
        default='"./output/llama-2-7b-chat-ci"',
    )
    args = parser.parse_args()
    LLAMA2_FINETUNEED_PATH = args.path

    interpreter = LlamaCodeInterpreter(
        model_path=LLAMA2_FINETUNEED_PATH,
        # load_in_4bit=True
    )

    sample_data = load_jsonl(JSON_PATH)
    correct = 0

    for idx, data_dict in enumerate(tqdm(sample_data[:100])):
        question = data_dict["question"]
        answer = data_dict["answer"]
        answer_only = data_dict["answer_refined"]

        # reset dilag : fewshot to follow answering format
        interpreter.dialog = [
            {
                "role": "system",
                "content": "You are helpful robot that can generate code , excute it and debug then answer",
            }
        ]  # this will replaced in template conversion
        # interpreter.dialog += few_shot_1
        # interpreter.dialog += few_shot_2
        # interpreter.dialog += few_shot_3

        output = interpreter.chat(
            user_message=f"{question}",
            VERBOSE=True,
        )

        pattern = r"\[RESULT_TOK\]\s*(\d+(\.\d+)?)\s*\[/RESULT_TOK\]"
        pred = -9212323
        if re.search(pattern, output["content"]):
            pred = re.search(pattern, output["content"]).group(1)

        pred = str(pred)
        answer_only = str(answer_only)
        if float(pred.replace(",", "")) == float(answer_only.replace(",", "")):
            correct += 1

        print("-" * 30)
        print(f"\tThe question was : {question}")
        print(f"\tThe answer was   : [{answer_only}]")
        print(f"\tModel pred is    : [{pred}]")
        print(f"\t    Accuracy     : [{correct/(idx+1)}]")
        print("-" * 30)
