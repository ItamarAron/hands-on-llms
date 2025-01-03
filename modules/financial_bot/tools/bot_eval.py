import logging
import json

import fire


from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truth": [ground_truth],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        #context_entity_recall,
        #context_recall,
        #context_relevancy,
        #context_utilization,
        faithfulness,
        answer_relevancy
    )

    metrics = [
        #context_utilization,
        answer_relevancy,
        #context_recall,
        answer_similarity,
        #context_entity_recall,
        #answer_correctness,
        faithfulness
    ]

    with open(testset_path, "r") as f:
        data = json.load(f)
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": elem["history"] if "history" in elem else [],
            }
            response = bot.answer(**input_payload)

            context = response["context"]
            logging.info('{"input": %s ,\n"context": %s \n"answer": %s',input_payload, context, response["answer"])
            logger.info("Score=%s", evaluate_w_ragas(query=elem["question"], context=context.split('\n'), output=response["answer"], ground_truth=elem["response"], metrics=metrics))

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
