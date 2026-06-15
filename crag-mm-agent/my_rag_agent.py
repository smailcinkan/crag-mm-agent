from typing import Dict, List, Any
import os

import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

from crag_web_result_fetcher import WebSearchResult
import vllm
from sentence_transformers import CrossEncoder

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2.
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1.
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1.
VLLM_TENSOR_PARALLEL_SIZE = 1
# VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_GPU_MEMORY_UTILIZATION = 0.75  # 降低显存占用率，为排序器模型留出空间

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
# NUM_SEARCH_RESULTS = 3
NUM_INITIAL_SEARCH_RESULTS = 10  # 初步检索时多召回一些结果
NUM_TOP_K_RERANKED_RESULTS = 5  # 排序后只选择最好的前5个


class MyRAGAgent(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your
    RAG submission for the CRAG-MM benchmark.
    Note: This implementation is not tuned for performance, and is intended for demonstration purposes only.

    This agent enhances responses by retrieving relevant information through a search pipeline
    and incorporating that context when generating answers. It follows a two-step approach:
    1. First, batch-summarize all images to generate effective search terms
    2. Then, retrieve relevant information and incorporate it into the final prompts

    The agent leverages batched processing at every stage to maximize efficiency.

    Note:
        This agent requires a search_pipeline for RAG functionality. Without it,
        the agent will raise a ValueError during initialization.

    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
            self,
            search_pipeline: UnifiedSearchPipeline,
            model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",  # <--- 修改这里
            max_gen_len: int = 64
    ):
        """
        Initialize the RAG agent with the necessary components.

        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
                      Hence, this implementation of the RAG agent is not suitable for Task 1 (Single-source Augmentation).
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.

        Raises:
            ValueError: If search_pipeline is None, as it's required for RAG functionality.
        """
        super().__init__(search_pipeline)

        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")

        self.model_name = model_name
        self.max_gen_len = max_gen_len

        self.initialize_models()

    def initialize_models(self):
        """
        Initialize the vLLM model and tokenizer with appropriate settings.

        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt, as
        Llama-3.2-Vision models do not handle multiple images well in a single prompt.

        Note:
            The limit_mm_per_prompt setting is critical as the current Llama vision models
            struggle with multiple images in a single conversation.
            Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        """
        print(f"Initializing {self.model_name} with vLLM...")

        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1
            }  # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()

        print("Models loaded successfully")
        # 在 initialize_models 函数的末尾添加
        print("Initializing Re-ranker model...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512, device='cuda')
        print("Re-ranker loaded successfully.")

    def _analyze_query_and_generate_search_term(self, query: str, image_summary: str) -> dict:
        """
        使用LLM分析查询，决定是否需要web搜索，并生成最佳搜索词。
        返回一个包含分析结果的字典。
        """
        print(f"INFO: Analyzing query: '{query[:30]}...'")

        # 设计一个专门用于任务规划的Prompt
        ANALYSIS_PROMPT_TEMPLATE = f"""You are a smart task router. Your job is to analyze a user's question and an image summary, 
        then decide the best action.
        Respond in JSON format with two keys: "requires_web_search" (true or false) and "search_query" (a string).
1.  If the question can be answered by just looking at the image (based on the summary), set "requires_web_search" to false 
    and "search_query" to an empty string.
    Examples: "What color is this?", "What is in the image?"
2.  If the question requires external factual knowledge (e.g., history, prices, ownership), set "requires_web_search" to true 
    and generate the best possible, concise search query.
    Examples: "Who made this painting?", "How much does this cost?"
---
Image Summary: "{image_summary}"
User Question: "{query}"
---
Your JSON response:"""

        analysis_messages = [{"role": "user", "content": ANALYSIS_PROMPT_TEMPLATE}]
        analysis_prompt_formatted = self.tokenizer.apply_chat_template(
            analysis_messages, add_generation_prompt=True, tokenize=False
        )

        # 调用LLM进行分析
        output = self.llm.generate(
            [analysis_prompt_formatted],
            sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=150, skip_special_tokens=True)
        )[0]

        response_text = output.outputs[0].text.strip()

        # 解析LLM返回的JSON结果
        try:
            import json
            # 找到JSON部分的开始和结束
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_string = response_text[json_start:json_end]
            analysis_result = json.loads(json_string)
            print(f"INFO: Analysis result: {analysis_result}")
            return analysis_result
        except Exception as e:
            print(f"ERROR: Failed to parse analysis JSON: {response_text}. Defaulting to web search. Error: {e}")
            # 如果解析失败，默认执行web搜索，以保证流程继续
            return {"requires_web_search": True, "search_query": image_summary}

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.

        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.

        Returns:
            int: The batch size, indicating how many queries should be processed together
                 in a single batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_summarize_images(self, queries: List[str], images: List[Image.Image]) -> List[str]:

        print("INFO: 正在生成与问题相关的图片摘要...")
        inputs = []

        # 新的、更智能的动态提示词模板
        SUMMARY_PROMPT_TEMPLATE = """You are an expert image analyst. Your task is to describe the image, 
                                    but specifically focus on details that will help answer the user's following question.
                                
                                    User's Question: "{query}"
                                
                                    Based on the user's question, provide a concise, 
                                    one-sentence description of the most relevant parts of the image.
                                    """

        for i in range(len(images)):
            query = queries[i]
            image = images[i]

            # 为每一个查询动态地构建提示词
            full_prompt_text = SUMMARY_PROMPT_TEMPLATE.format(query=query)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_prompt_text}]}]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})

        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(temperature=0.0,
                                                max_tokens=40,
                                                skip_special_tokens=True)
        )

        summaries = [output.outputs[0].text.strip() for output in outputs]
        print(f"已生成 {len(summaries)} 条与问题相关的图片摘要。")
        return summaries

    def prepare_rag_enhanced_inputs(
            self,
            queries: List[str],
            images: List[Image.Image],
            image_summaries: List[str],
            message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:

        print("INFO: Combining original query and image summary for a better search query.")
        search_queries = [f"{q} {s}" for q, s in zip(queries, image_summaries)]


        # --- (后续的初步检索、排序、最终生成等逻辑保持不变) ---
        NUM_INITIAL_SEARCH_RESULTS = 10
        NUM_TOP_K_RERANKED_RESULTS = 3

        # --- (构造搜索词) ---
        # search_queries = image_summaries

        # --- (这部分也和之前一样，是初步检索) ---
        search_results_batch = []
        for i, search_query in enumerate(search_queries):
            # 为了给排序器提供更多选择，我们可以让检索器返回更多的结果，比如10个
            # results = self.search_pipeline(search_query, k=10)
            results = self.search_pipeline(search_query, k=NUM_INITIAL_SEARCH_RESULTS)
            search_results_batch.append(results)

        # ▼▼▼▼▼▼▼▼▼ 新增：初始化排序器模型 ▼▼▼▼▼▼▼▼▼
        # 为避免重复加载，可以将这行移动到类的__init__方法中
        reranker_model = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        inputs = []
        for idx, (query, image, message_history, search_results) in enumerate(
                zip(queries, images, message_histories, search_results_batch)
        ):
            # --- ▼▼▼▼▼ 新增的调试代码 ▼▼▼▼▼ ---
            # 我们只对这个特定的失败案例进行详细打印
            if "who made this painting" in query:
                print("\n\n--- DEBUGGING CASE: 'who made this painting' ---")

                # 1. 打印图片摘要和搜索词
                print(f"DEBUG[1]: Image Summary: '{image_summaries[idx]}'")
                print(f"DEBUG[2]: Search Query used: '{search_queries[idx]}'")

                # 3. 打印初步检索到的Top-3结果
                print(f"DEBUG[3]: Top 3 Initial Retrieval Results:")
                if search_results:
                    for i, res in enumerate(search_results[:3]):
                        print(f"  - Result {i + 1}: {res.get('page_snippet', 'N/A')}")
                else:
                    print("  - No initial results found.")

                # 4. 打印排序后的Top-3结果 (如果您已加入排序器)
                # (如果您还没有加入排序器，可以先注释掉下面这部分)
                if 'top_k_results' in locals() and top_k_results:
                    print(f"DEBUG[4]: Top 3 Re-ranked Results:")
                    for i, res in enumerate(top_k_results):
                        print(f"  - Reranked {i + 1}: {res.get('page_snippet', 'N/A')}")
                else:
                    print("  - No re-ranked results available.")

                # 5. 打印最终喂给模型的RAG上下文
                # (为了看到最终的rag_context, 您需要把这段代码移动到rag_context构造完成之后)
                # 暂时我们先不加这个，避免代码结构混乱
                print("--------------------------------------------------\n\n")
            # --- ▲▲▲▲▲ 调试代码结束 ▲▲▲▲▲ ---
            # --- ▼▼▼▼▼ 新增：排序逻辑 ▼▼▼▼▼ ---
            #新的排序逻辑
            if search_results:
                # 1. 准备排序数据：(问题, 文档片段) 对
                pairs = [[query, result.get('page_snippet', '')] for result in search_results]
                # 2. 使用已加载的排序器模型进行打分
                print(f"Re-ranking {len(pairs)} snippets for query: '{query[:30]}...'")
                scores = self.reranker.predict(pairs)
                # 3. 将分数和原始结果打包，并按分数从高到低排序
                scored_results = sorted(zip(scores, search_results), key=lambda x: x[0], reverse=True)
                # 4. 只选择分数最高的Top-K个结果
                top_k_results = [result for score, result in scored_results[:NUM_TOP_K_RERANKED_RESULTS]]
                print(f"Selected Top-{len(top_k_results)} results after re-ranking.")
            else:
                top_k_results = []
            #排序逻辑结束
            # --- ▲▲▲▲▲ 排序逻辑结束 ▲▲▲▲▲ ---

            # --- (这部分和之前一样，是构造最终Prompt，但现在用的是排序后的结果) ---
            SYSTEM_PROMPT = (
                "You are a factual and precise assistant. Your ONLY task is to answer the user's question based *ONLY* on the provided image and the context snippets given below, prefixed with [Info].\n"
                "You are forbidden from using any of your own internal knowledge. Do not make up information.\n"
                "If the answer cannot be found in the provided context, you MUST ONLY respond with the exact phrase 'I don't know'."
            )

            rag_context = ""
            # 注意：这里的循环现在使用的是 top_k_results
            if top_k_results:
                rag_context = "Here is the context retrieved and re-ranked from a web search. Use this information to answer the question:\n\n"
                for i, result in enumerate(top_k_results):
                    result = WebSearchResult(result)
                    snippet = result.get('page_snippet', '')
                    if snippet:
                        rag_context += f"[Info {i + 1}] {snippet}\n\n"
            else:
                rag_context = "No relevant information was found after retrieval and re-ranking.\n"

            # ... (后续构造messages和formatted_prompt的逻辑保持不变) ...
            messages = [{"role": "user", "content": [{"type": "image"}]}]
            if message_history:
                messages.extend(message_history)
            full_user_prompt = f"{SYSTEM_PROMPT}\n\n{rag_context}Based on the image and the context above, answer the following question:\n{query}"
            messages.append({"role": "user", "content": full_user_prompt})

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})

        return inputs

    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        print(f"Processing batch of {len(queries)} with a new 'Self-Correction' RAG strategy.")
        # 步骤一：生成初步草稿
        print("Step 1: Generating initial draft answers...")
        # 回归到最简单有效的“仅图片摘要”搜索策略
        image_summaries = self.batch_summarize_images(queries, images)
        # 获取用于生成草稿的输入（包含上下文）
        draft_inputs_with_context = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )
        # 提取纯文本的prompt用于生成
        draft_prompts = [item["prompt"] for item in draft_inputs_with_context]
        draft_outputs = self.llm.generate(
            draft_prompts,
            sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=MAX_GENERATION_TOKENS,
                                                skip_special_tokens=True)
        )
        draft_answers = [output.outputs[0].text.strip() for output in draft_outputs]
        print("Successfully generated draft answers.")
        # ------------------------------------------------------------------
        # 步骤二和三：对每个草稿进行“反思-修正”循环
        # ------------------------------------------------------------------
        print("Step 2 & 3: Reflecting on and revising draft answers...")
        final_responses = []
        for i in range(len(queries)):
            original_query = queries[i]
            draft_answer = draft_answers[i]
            # 从我们之前准备好的输入中，提取出给模型参考的RAG上下文
            rag_context_prompt = draft_inputs_with_context[i]["prompt"]

            # 设计一个用于“反思”和“修正”的、非常严格的最终Prompt
            REVISION_PROMPT_TEMPLATE = f"""You are a meticulous fact-checker and editor.
    I will provide you with a user's question, a set of reference materials (context), and a pre-generated "draft answer".
    Your task is to analyze the draft answer and generate a final, corrected answer based on these strict rules:

    1.  Read the user's question and the reference context carefully.
    2.  Critically evaluate the "draft answer". Does it accurately reflect the information found ONLY in the reference context?
    3.  If the draft answer is fully supported by the context, use it as the final answer.
    4.  If the draft answer contains any information, details, or claims NOT present in the reference context, it is a hallucination. In this case, your final answer MUST be the exact phrase 'I don't know'.
    5.  If the draft answer is a polite refusal (e.g., "I'm sorry...", "I cannot answer..."), your final answer MUST also be the exact phrase 'I don't know'.

    ---
    ### User's Question:
    {original_query}

    ### Reference Context:
    {rag_context_prompt}

    ### Draft Answer to Revise:
    "{draft_answer}"
    ---

    ### Final, Corrected Answer:
    """

            # 这里我们只把修正指令发给模型，不需要再传图片
            revision_messages = [{"role": "user", "content": REVISION_PROMPT_TEMPLATE}]
            revision_prompt_formatted = self.tokenizer.apply_chat_template(
                revision_messages, add_generation_prompt=True, tokenize=False
            )

            # 调用LLM进行最终的修正生成
            final_output = self.llm.generate(
                [revision_prompt_formatted],
                sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=MAX_GENERATION_TOKENS,
                                                    skip_special_tokens=True)
            )[0]

            final_responses.append(final_output.outputs[0].text.strip())

        print(f"Successfully generated {len(final_responses)} final responses after revision.")
        return final_responses