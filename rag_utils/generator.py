# ----------------------------------------------------------------------------
# NOTICE: This code is the exclusive property of Cornell University
#         Computer Architecture Research and is strictly confidential.
#
#         Unauthorized distribution, reproduction, or use of this code, in
#         whole or in part, is strictly prohibited. This includes, but is
#         not limited to, any form of public or private distribution,
#         publication, or replication.
#
# For inquiries or access requests, please contact:
#         Zuoming Fu (zf242@cornell.edu)
#
# Reference:
#       SystemServant by Alex Manley (amanley97@ku.edu)
# ----------------------------------------------------------------------------

from langchain_openai import ChatOpenAI

import torch
from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model
from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer
from langchain_core.messages.ai import AIMessage
import transformers


# ----------------------------------------------------------------------------
# Generator Using OpenAI API
# ----------------------------------------------------------------------------

class OpenAIGenerator:

    def __init__(self, model_name="gpt-4o-mini", **kwargs):
        '''Create a generator object.
        
        Parameters:
        model_name (str): The model name to use for generation.
        source (str): The source of the model.
        '''

        self.llm = ChatOpenAI(model=model_name)
        self.kwargs = kwargs

    def gen_resp(self, message):
        '''Generate a response to a message.
        
        Parameters:
        message (str): The message to generate a response to.
        '''

        response = self.llm.invoke(message)
        return response

# ----------------------------------------------------------------------------
# Generator Using SystemServant (deprecated)
# ----------------------------------------------------------------------------

class SystemSavantModel:
    def __init__(
        self,
        model_name,
        peft_model: str = None,
        quantization: str = None,  # Options: 4bit, 8bit
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions
        top_p: float = 1.0,  # [optional] Top-p filtering
        temperature: float = 1.0,  # [optional] Modulation of next token probabilities.
        top_k: int = 50,  # [optional] Top-k filtering
        repetition_penalty: float = 1.0,  # [optional] Repetition penalty
        length_penalty: int = 1,  # [optional] Penalty for length in beam generation.
        enable_azure_content_safety: bool = False,  # Enable Azure content safety
        enable_sensitive_topics: bool = False,  # Check for sensitive topics
        enable_salesforce_content_safety: bool = True,  # Enable Salesforce safety check
        enable_llamaguard_content_safety: bool = False,  # Enable LlamaGuard content safety
        max_padding_length: int = None,  # Max padding length for tokenizer
        use_fast_kernels: bool = False,  # Enable SDPA for memory-efficient kernels
        share_gradio: bool = False,  # Enable Gradio sharing
        **kwargs,
    ):
        # Store the provided arguments as attributes
        self.model_name = model_name
        self.peft_model = peft_model
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.prompt_file = prompt_file
        self.seed = seed
        self.do_sample = do_sample
        self.min_length = min_length
        self.use_cache = use_cache
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.enable_azure_content_safety = enable_azure_content_safety
        self.enable_sensitive_topics = enable_sensitive_topics
        self.enable_salesforce_content_safety = enable_salesforce_content_safety
        self.enable_llamaguard_content_safety = enable_llamaguard_content_safety
        self.max_padding_length = max_padding_length
        self.use_fast_kernels = use_fast_kernels
        self.share_gradio = share_gradio
        self.kwargs = kwargs

        # Setup the model and tokenizer
        self._setup_seed()
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _setup_seed(self):
        print(f"[debug] setting seed: {self.seed}")
        if is_xpu_available():
            torch.xpu.manual_seed(self.seed)
        else:
            torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

    def _load_model(self):
        return load_model(
            self.model_name, self.quantization, self.use_fast_kernels, **self.kwargs
        )

    def gen_resp(self, user_prompt, temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=200):

        if not isinstance(user_prompt, str):
            user_prompt = user_prompt.text

        safety_checker = get_safety_checker(
            self.enable_azure_content_safety,
            self.enable_sensitive_topics,
            self.enable_salesforce_content_safety,
            self.enable_llamaguard_content_safety,
        )

        safety_results = [check(user_prompt) for check in safety_checker]
        are_safe = all([r[1] for r in safety_results])
        if not are_safe:
            print("Skipping the inference as the prompt is not safe.")
            return


        batch = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_padding_length,
            return_tensors="pt",
        )

        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,  # Use the dynamic value passed from Gradio
                do_sample=self.do_sample,
                top_p=top_p,  # Use the dynamic value passed from Gradio
                temperature=temperature,  # Use the dynamic value passed from Gradio
                min_length=self.min_length,
                use_cache=self.use_cache,
                top_k=top_k,  # Use the dynamic value passed from Gradio
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                eos_token_id=128009,
                **self.kwargs,
            )


        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        safe_output = self.safety_check_output(user_prompt, output_text, safety_checker)
        non_verbose_output = safe_output.split("user")[-1].strip()[len(user_prompt):]

        message = AIMessage(non_verbose_output)

        return message

    def safety_check_output(self, prompt, output, checker):
        safety_results = [
            check(output, agent_type=AgentType.AGENT, user_prompt=prompt)
            for check in checker
        ]
        are_safe = all([r[1] for r in safety_results])
        if are_safe:
            # print("User input and model output deemed safe.")
            # print(f"Model output:\n{output}")
            return output
        else:
            # print("Model output deemed unsafe.")
            # for method, is_safe, report in safety_results:
            #     if not is_safe:
            #         print(method)
            #         print(report)
            return None