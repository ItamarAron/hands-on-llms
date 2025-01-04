import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableSequence
# ADDED IMPORT
from langchain_openai import ChatOpenAI

from financial_bot import constants
from financial_bot.chains import (
    CompressHistoryChain,
    ContextExtractorChain,
    FinancialBotQAChain, compressed_history_key, history_input_key,
)
from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.handlers import CometLLMMonitoringHandler
from financial_bot.models import build_huggingface_pipeline
from financial_bot.qdrant import build_qdrant_client
from financial_bot.template import get_llm_template

# set_verbose(True)
# set_debug(True)

logger = logging.getLogger(__name__)


class FinancialBot:
    """
    A language chain bot that uses a language model to generate responses to user inputs.

    Args:
        llm_model_id (str): The ID of the Hugging Face language model to use.
        llm_qlora_model_id (str): The ID of the Hugging Face QLora model to use.
        llm_template_name (str): The name of the LLM template to use.
        llm_inference_max_new_tokens (int): The maximum number of new tokens to generate during inference.
        llm_inference_temperature (float): The temperature to use during inference.
        vector_collection_name (str): The name of the Qdrant vector collection to use.
        vector_db_search_topk (int): The number of nearest neighbors to search for in the Qdrant vector database.
        model_cache_dir (Path): The directory to use for caching the language model and embedding model.
        streaming (bool): Whether to use the Hugging Face streaming API for inference.
        embedding_model_device (str): The device to use for the embedding model.
        debug (bool): Whether to enable debug mode.

    Attributes:
        finbot_chain (Chain): The language chain that generates responses to user inputs.
    """

    def __init__(
            self,
            llm_model_id: str = constants.LLM_MODEL_ID,
            llm_qlora_model_id: str = constants.LLM_QLORA_CHECKPOINT,
            llm_template_name: str = constants.TEMPLATE_NAME,
            llm_inference_max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
            llm_inference_temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
            vector_collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
            vector_db_search_topk: int = constants.VECTOR_DB_SEARCH_TOPK,
            model_cache_dir: Path = constants.CACHE_DIR,
            streaming: bool = False,
            embedding_model_device: str = "cuda:0",
            debug: bool = False,
    ):
        self._llm_model_id = llm_model_id
        self._llm_qlora_model_id = llm_qlora_model_id
        self._llm_template_name = llm_template_name
        self._llm_template = get_llm_template(name=self._llm_template_name)
        self._llm_inference_max_new_tokens = llm_inference_max_new_tokens
        self._llm_inference_temperature = llm_inference_temperature
        self._vector_collection_name = vector_collection_name
        self._vector_db_search_topk = vector_db_search_topk
        self._debug = debug

        self._qdrant_client = build_qdrant_client()

        self._embd_model = EmbeddingModelSingleton(
            cache_dir=model_cache_dir, device=embedding_model_device
        )
        self._llm_agent, self._streamer = build_huggingface_pipeline(
            llm_model_id=llm_model_id,
            llm_lora_model_id=llm_qlora_model_id,
            max_new_tokens=llm_inference_max_new_tokens,
            temperature=llm_inference_temperature,
            use_streamer=streaming,
            cache_dir=model_cache_dir,
            debug=debug,
        )
        self.finbot_chain = self.build_chain()

    @property
    def is_streaming(self) -> bool:
        return self._streamer is not None

    def build_chain(self) -> Runnable:
        """
        Constructs and returns a financial bot chain.
        This chain is designed to take as input the user description, `about_me` and a `question` and it will
        connect to the VectorDB, searches the financial news that rely on the user's question and injects them into the
        payload that is further passed as a prompt to a financial fine-tuned LLM that will provide answers.

        The chain consists of two primary stages:
        1. Context Extractor: This stage is responsible for embedding the user's question,
        which means converting the textual question into a numerical representation.
        This embedded question is then used to retrieve relevant context from the VectorDB.
        The output of this chain will be a dict payload.

        2. LLM Generator: Once the context is extracted,
        this stage uses it to format a full prompt for the LLM and
        then feed it to the model to get a response that is relevant to the user's question.

        Returns
        -------
        chains.SequentialChain
            The constructed financial bot chain.

        Notes
        -----
        The actual processing flow within the chain can be visualized as:
        [about: str][question: str] > ContextChain >
        [about: str][question:str] + [context: str] > FinancialChain >
        [answer: str]
        """

        llm = ChatOpenAI(model_name="gpt-4o-mini")
        json_llm: Runnable = llm.bind(response_format={"type": "json_object"})
        rephrase_template: Runnable = PromptTemplate.from_template("Question: {question} "
                                                                   "Given the above question, rephrase and expand it to help you do better answering."
                                                                   "Maintain all information in the original question."
                                                                   "give 3 different options."
                                                                   "return JSON object with key 'rephrased_questions' and a value with array of the rephrased questions.")

        rephrase_question_chain = rephrase_template | json_llm | JsonOutputParser()

        context_retrieval_chain = ContextExtractorChain(
            embedding_model=self._embd_model,
            vector_store=self._qdrant_client,
            vector_collection=self._vector_collection_name,
            top_k=self._vector_db_search_topk,
        )

        logger.info("Building 3/4 - FinancialBotQAChain")
        callbacks = self.get_callbacks()

        composite_generate_chain = {"answer": FinancialBotQAChain(hf_pipeline=self._llm_agent,
                                                                  template=self._llm_template,
                                                                  callbacks=callbacks
                                                                  ) | (lambda x: x["answer"]),
                                    "question": lambda x: x["context"]["question"],
                                    "context": lambda x: x["context"]["context"]}

<<<<<<< Updated upstream
        def format_history(input:  dict) -> dict:
            history = [f"Question: {human}\n Answer: {ai}" for human, ai in input[history_input_key]]
            return {"history": history}

  ## As recommended by: https://devblogs.microsoft.com/surface-duo/android-openai-chatgpt-18/
        summarize_history_template  = PromptTemplate.from_template(
        """Summarize the following conversation and extract key points:
            ####
            {history}
            ####""")
=======
        def format_history(input: dict) -> dict:
            history = [f"Question: {human}\n Answer: {ai}" for human, ai in input[history_input_key]]
            return {"history": history}

        ## As recommended by: https://devblogs.microsoft.com/surface-duo/android-openai-chatgpt-18/
        summarize_history_template = PromptTemplate.from_template(
            """Summarize the following conversation and extract key points:
                ####
                {history}
                ####""")
>>>>>>> Stashed changes

        summarize_history_chain = format_history | summarize_history_template | llm | StrOutputParser()

        preparation_chain = {compressed_history_key: summarize_history_chain,
<<<<<<< Updated upstream
                             "context": context_retrieval_chain,
                             "rephrased_questions": rephrase_question_chain}
=======
                             "context": context_retrieval_chain, }
        # "rephrased_questions": rephrase_question_chain}
>>>>>>> Stashed changes

        pick_best_template = (
            'Given this question: "{question}"'
            "From the answers below, read them carefully and choose the one that might be considered the best overall."
            "Answer #1:\n {answer_0}\n"
            "Answer #2:\n {answer_1}\n"
            "Answer #3:\n {answer_2}\n"
            "Your output should be only the text of the chosen answer and nothing else"
        )

        choose_response_chain = {"answer": PromptTemplate.from_template(pick_best_template) | llm | StrOutputParser(),
                                 "context": lambda x: x["context"]}

        seq_chain = RunnableSequence(preparation_chain, composite_generate_chain)#, choose_response_chain)

        logger.info("seq_chain: %s", seq_chain)
        logger.info("Done building SequentialChain.")
        logger.info("Workflow:")
        logger.info(
            """
            [about: str][question: str] > ContextChain > 
            [about: str][question:str] + [context: str] > FinancialChain > 
            [answer: str]
            """
        )
        return seq_chain

    def get_callbacks(self):
        if self._debug:
            return []
        else:
            try:
                comet_project_name = os.environ["COMET_PROJECT_NAME"]
            except KeyError:
                raise RuntimeError(
                    "Please set the COMET_PROJECT_NAME environment variable."
                )
            return [
                CometLLMMonitoringHandler(
                    project_name=f"{comet_project_name}-monitor-prompts",
                    llm_model_id=self._llm_model_id,
                    llm_qlora_model_id=self._llm_qlora_model_id,
                    llm_inference_max_new_tokens=self._llm_inference_max_new_tokens,
                    llm_inference_temperature=self._llm_inference_temperature,
                )
            ]

    def answer(
            self,
            about_me: str,
            question: str,
            to_load_history: List[Tuple[str, str]] = None,
    ) -> str:
        """
        Given a short description about the user and a question make the LLM
        generate a response.

        Parameters
        ----------
        about_me : str
            Short user description.
        question : str
            User question.

        Returns
        -------
        str
            LLM generated response.
        """

        inputs = {
            "about_me": about_me,
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.finbot_chain.invoke(inputs)
        logger.info("financial bot response: %s", response)
        return response

    def stream_answer(self) -> Iterable[str]:
        """Stream the answer from the LLM after each token is generated after calling `answer()`."""

        assert (
            self.is_streaming
        ), "Stream answer not available. Build the bot with `use_streamer=True`."

        partial_answer = ""
        for new_token in self._streamer:
            if new_token != self._llm_template.eos:
                partial_answer += new_token

                yield partial_answer
