from collections import Counter
from itertools import combinations
from math import comb


def evaluate_item(predictions, gold_label):
    """
    predictions:
        Example: ["B", "B", "B", "C", "D"]

    gold_label:
        Example: "B"
    """

    m = len(predictions)

    if m < 2:
        raise ValueError("At least two prompt predictions are required.")

    # -----------------------------------
    # 1. Correct / Incorrect
    # -----------------------------------

    correctness = [
        int(pred == gold_label)
        for pred in predictions
    ]

    num_correct = sum(correctness)
    num_wrong = m - num_correct

    prompt_accuracy = num_correct / m

    # -----------------------------------
    # 2. PSS
    # -----------------------------------

    pair_differences = []

    for a, b in combinations(correctness, 2):
        pair_differences.append(abs(a - b))

    pss = sum(pair_differences) / len(pair_differences)

    # -----------------------------------
    # 3. Answer distribution
    # -----------------------------------

    counts = Counter(predictions)

    total_pairs = comb(m, 2)

    same_answer_pairs = sum(
        comb(count, 2)
        for count in counts.values()
        if count >= 2
    )

    agreement = same_answer_pairs / total_pairs

    disagreement = 1 - agreement

    # -----------------------------------
    # 4. Our candidate metric: GAPR
    # -----------------------------------

    gapr = prompt_accuracy * agreement

    # -----------------------------------
    # 5. Strict Robust Accuracy
    # -----------------------------------

    all_correct = int(num_correct == m)

    # -----------------------------------
    # 6. Voting
    # -----------------------------------

    sorted_counts = counts.most_common()

    top_answer = sorted_counts[0][0]
    top_count = sorted_counts[0][1]

    tie = (
        len(sorted_counts) > 1
        and sorted_counts[0][1] == sorted_counts[1][1]
    )

    if tie:
        vote_prediction = None
        vote_correct = None
    else:
        vote_prediction = top_answer
        vote_correct = int(top_answer == gold_label)

    # -----------------------------------
    # 7. Wrong Consensus
    # -----------------------------------

    consensus_ratio = top_count / m

    wrong_consensus = int(
        not tie
        and top_answer != gold_label
        and consensus_ratio >= 0.8
    )

    return {
        "gold_label": gold_label,

        "predictions": predictions,
        "correctness": correctness,

        "num_correct": num_correct,
        "num_wrong": num_wrong,

        "prompt_accuracy": prompt_accuracy,

        "pss": pss,

        "answer_agreement": agreement,
        "answer_disagreement": disagreement,

        "gapr": gapr,

        "all_correct": all_correct,

        "vote_prediction": vote_prediction,
        "vote_correct": vote_correct,

        "consensus_ratio": consensus_ratio,
        "wrong_consensus": wrong_consensus,
    }