import re
import json
import argparse
from typing import List, Dict

from tqdm import tqdm
import numpy as np
import backoff
import openai
from scipy.stats import pearsonr, spearmanr, kendalltau

openai.api_key = <YOUR_OPENAI_KEY>

PROMPT = "You will be given one summary written for a news article.\n" \
         "Your task is to rate the summary on one metric.\n\n" \
         "Evaluation Criteria:\n" \
         "Relevance (1-5) The rating measures how well the summary captures the key points of the article. " \
         "Consider whether all and only the important aspects are contained in the summary.\n\n"\
         "News article:\n" \
         "[article]\n" \
         "Summary:\n" \
         "[summary]\n\n" \
         "Based on the news article and the evaluation criteria for relevance, please rate the relevance " \
         "of the summary.\n" \
         "Relevance Score:"


def load_dataset(file_path):
    return json.load(open(file_path, "r", encoding="utf8"))


def correlation(pred_scores, gold_scores):
    r_pearsonr, p_pearsonr = pearsonr(pred_scores, gold_scores)
    r_spearmanr, p_spearmanr = spearmanr(pred_scores, gold_scores)
    r_kendalltau, p_kendalltau = kendalltau(pred_scores, gold_scores)

    pearsonr_res = str(np.round(r_pearsonr, 3)) + ' (' + str(np.round(p_pearsonr, 3)) + ')'
    spearmanr_res = str(np.round(r_spearmanr, 3)) + ' (' + str(np.round(p_spearmanr, 3)) + ')'
    kendalltau_res = str(np.round(r_kendalltau, 3)) + ' (' + str(np.round(p_kendalltau, 3)) + ')'
    return pearsonr_res, spearmanr_res, kendalltau_res


def generate_prompt(article, summary):
    prompt = PROMPT.replace("[article]", article)
    prompt = prompt.replace("[summary]", summary)
    #print(prompt)
    return prompt


def extract_first_number_from_string(string):
    pattern = r"\d+(\.\d+)?"
    match = re.search(pattern, string)
    if match:
        number = match.group()
        return float(number)
    else:
        return None


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def compute_score(article, summary, args):
    prompt = generate_prompt(article, summary)
    response = openai.ChatCompletion.create(
        model=args.model_name,
        messages=[
            {"role": "user",
             "content": prompt},
        ],
        temperature=args.temperature,
        n=args.n
    )
    prompt += response["choices"][0]["message"]["content"]
    # print(prompt)
    score = float(extract_first_number_from_string(response["choices"][0]["message"]["content"]))
    return score, prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=int)
    parser.add_argument("--model_name", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()
    print(args)

    file_path = f"data/summeval.json"
    dataset = load_dataset(file_path)
    print(f"Load summeval, size={len(dataset)}")

    scores = []
    prompts = []
    golds = []
    for example in tqdm(dataset):
        article = example['source'].strip().strip('\n')
        summary = example['system_output'].strip().strip('\n')
        while True:
            try:
                score, prompt = compute_score(article, summary, args)
                break
            except Exception as e:
                print(e)
        scores.append(score)
        prompts.append(prompt)
        golds.append(example['scores']['relevance'])

    pearson, spearman, kendalltau = correlation(scores, golds)
    print(f"pearson: {pearson}\nspearman: {spearman}\nkendalltau: {kendalltau}")

    result = []
    for ps, gs, prompt in zip(scores, golds, prompts):
        result.append({
            "prompt": prompt,
            "score": ps,
            "gold_score": gs
        })
    json.dump(result, open(f"summeval/output/origin/{args.output}.json", "w", encoding="utf8"))

    with open(f"summeval/output/origin/{args.output}.txt", "w", encoding="utf8") as fw:
        for ps, gs in zip(scores, golds):
            fw.write(f"{ps}\t{gs}\n")
