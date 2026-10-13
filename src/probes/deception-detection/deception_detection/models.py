from enum import Enum
from functools import cache
from pathlib import Path
from typing import Literal, cast, overload

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from transformers import PreTrainedTokenizerBase

from deception_detection.log import logger
from nnsight import LanguageModel, VisionLanguageModel


# We only use the instruct variations of the models. Note not all sizes are available for all model
# families.
ModelType = Literal["gemma", "mistral", "llama", "qwen"]
ModelSize = Literal[1, 2, 3, 7, 8, 9, 24, 27, 70, 72]


class ModelName(str, Enum):
    MISTRAL_3_1_24B = "mistral-small-3.1-24b"
    LLAMA_70B_3_3 = "llama-70b-3.3"
    LLAMA_70B_3_1 = "llama-70b-3.1"
    QWEN_72B = "qwen-72b"
    GEMMA_3_27B = "gemma-3-27b"

    @property
    def type(self) -> ModelType:
        return cast(ModelType, self.value.split("-")[0])

    @property
    def size(self) -> ModelSize:
        if self == ModelName.MISTRAL_3_1_24B:
            return cast(ModelSize, 24)
        elif self == ModelName.QWEN_72B:
            return cast(ModelSize, 72)
        elif self == ModelName.GEMMA_3_27B:
            # the version number, not the size, is the second field of "gemma-3-27b"
            return cast(ModelSize, 27)
        else:
            size_str = self.value.split("-")[1].replace("b", "")
            return cast(ModelSize, int(size_str))

    @property
    def is_llama_3_3(self) -> bool:
        return self == ModelName.LLAMA_70B_3_3
    
    @property
    def is_llama_3_1(self) -> bool:
        return self == ModelName.LLAMA_70B_3_1

    @property
    def is_mistral(self) -> bool:
        return self == ModelName.MISTRAL_3_1_24B
    
    @property
    def is_qwen(self) -> bool:
        return self == ModelName.QWEN_72B

    @property
    def is_vlm(self) -> bool:
        """A multimodal HF class (Mistral3 / Gemma3ForConditionalGeneration), which NDIF serves as
        an nnsight VisionLanguageModel and whose decoder blocks sit under `language_model`."""
        return self in (ModelName.MISTRAL_3_1_24B, ModelName.GEMMA_3_27B)


MODEL_REPOS: dict[ModelName, str] = {
    ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
    ModelName.LLAMA_70B_3_1: "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ModelName.MISTRAL_3_1_24B: "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    ModelName.QWEN_72B: "Qwen/Qwen2.5-72B-Instruct",
    ModelName.GEMMA_3_27B: "google/gemma-3-27b-it",
}

# Tokenizer and chat-template revisions, pinned so a later upstream edit cannot change the tokens
# a run reads (checked 2026-10-09; each is the repo's main at that date). Llama keeps loading main,
# as every existing Llama run did. Only the tokenizer is pinned: passing a revision to the nnsight
# constructor would change its remote model key, and the request would miss NDIF's deployment.
TOKENIZER_REVISIONS: dict[ModelName, str] = {
    ModelName.MISTRAL_3_1_24B: "68faf511d618ef198fef186659617cfd2eb8e33a",
    ModelName.QWEN_72B: "495f39366efef23836d0cfae4fbe635880d2be31",
    ModelName.GEMMA_3_27B: "005ad3404e59d6023443cb575daa05336842228a",
}

# Mistral's chat template builds its default system prompt from the current date. Only
# dialogues without a system message take that default -- RepE here -- so its tokens would
# otherwise change every day. `pin_template_date` swaps the call for a fixed date.
MISTRAL_DATE_CALL = 'strftime_now("%Y-%m-%d")'


def pin_template_date(template: str, date: str) -> str:
    """`template` with its single `strftime_now("%Y-%m-%d")` replaced by the literal `date`."""
    found = template.count(MISTRAL_DATE_CALL)
    if found != 1:
        raise ValueError(f"expected one {MISTRAL_DATE_CALL} in the chat template, found {found}")
    return template.replace(MISTRAL_DATE_CALL, f'"{date}"')


def prune_to_decoder_blocks(model) -> None:
    """Cut an nnsight VLM wrapper's envoy tree down to the chain a block read uses:
    `model.model.language_model.layers` and the blocks themselves, without their insides.

    A remote request pickles the wrapper's whole envoy tree, each module as a reference to its
    path, and NDIF resolves every path on its own copy of the model. Gemma-3's tree differs
    between the local transformers (4.55) and NDIF's (5.8) -- its vision tower is nested
    differently and its rotary embeddings were restructured -- so a request failed with "Unknown
    persistent id" for each such path. Reads touch only a block's output, so the pruned modules
    are never referenced; the meta modules themselves stay, and the server runs its own model.
    The trace-time LoRA, which needs each block's projections, is not used with these models.
    """
    from nnsight.intervention.envoy import Envoy

    def keep_only(envoy, names: set[str]) -> None:
        for key in [k for k, v in envoy.__dict__.items() if isinstance(v, Envoy) and k not in names]:
            del envoy.__dict__[key]

    keep_only(model.model, {"language_model"})
    keep_only(model.model.language_model, {"layers"})
    for block in model.model.language_model.layers:
        keep_only(block, set())


def get_model_name(model_name: ModelName) -> str:
    return model_name.value


def get_num_layers_for_model(model_name: ModelName) -> int:
    if model_name.type == "llama":
        n_layers = {70: 80}
    elif model_name.type == "mistral":
        n_layers = {24: 40}  # Mistral-Small-3.1's text model (text_config.num_hidden_layers)
    elif model_name.type == "qwen":
        n_layers = {72: 80}
    elif model_name.type == "gemma":
        n_layers = {27: 62}  # Gemma-3-27B's text model (text_config.num_hidden_layers)
    else:
        raise NotImplementedError(f"No number of layers for model {model_name}")
    return n_layers[model_name.size]


def load_tokenizer(
    model_name: ModelName, template_date: str | None = None
) -> PreTrainedTokenizerBase:
    """The tokenizer the pipeline tokenises with, set up as the authors' code sets it up.

    Mistral and Gemma take the processor's tokenizer with the processor's chat template copied
    over (their tokenizers have none of their own); Qwen's BOS is its EOS; every model pads on
    the left with BOS. `template_date` pins Mistral's default-system-prompt date.
    """
    model_name = ModelName(model_name)  # callers pass the enum or its string value
    repo = MODEL_REPOS[model_name]
    revision = TOKENIZER_REVISIONS.get(model_name)
    if model_name.is_vlm:
        processor = AutoProcessor.from_pretrained(repo, padding_side="left", revision=revision)
        tokenizer = processor.tokenizer
        if getattr(processor, "chat_template", None):
            tokenizer.chat_template = processor.chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left", revision=revision)
    if model_name.is_qwen:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if template_date is not None:
        if not model_name.is_mistral:
            raise ValueError(f"template_date only applies to Mistral, not {model_name.value}")
        tokenizer.chat_template = pin_template_date(tokenizer.chat_template, template_date)
    # every unpadding step (acts_cache.unpad_left, tail lengths) assumes left padding
    tokenizer.padding_side = "left"
    return tokenizer


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[True],
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
    template_date: str | None = None,
) -> tuple[None, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: Literal[False] = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
    template_date: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]: ...


@overload
def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
    template_date: str | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]: ...


def get_model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    pw_locked: bool = False,
    lora_path: str | Path | None = None,
    cut_at_layer: int | None = None,
    template_date: str | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    """
    Load the model and tokenizer based on the given parameters.
    """
    model_name = ModelName(model_name)  # callers pass the enum or its string value
    model, tokenizer = model_and_tokenizer(model_name, omit_model, lora_path, template_date)

    if cut_at_layer is not None and model is not None:
        # Handle PeftModel (LoRA) - need to access base model
        if isinstance(model, PeftModel):
            base_model = model.base_model.model
        else:
            base_model = model
            
        if hasattr(base_model.model, "layers"):                  # Llama-2/3, Gemma
            base_model.model.layers = base_model.model.layers[:cut_at_layer]
        elif hasattr(base_model.model, "language_model"):          # Mistral-3 family
            base_model.model.language_model.layers = base_model.model.language_model.layers[:cut_at_layer]
        else:
            raise ValueError(f"Model {model_name} has no layers to cut at layer {cut_at_layer}")

    return model, tokenizer


@cache
def get_llama3_tokenizer(tokenizer_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


def model_and_tokenizer(
    model_name: ModelName,
    omit_model: bool = False,
    lora_path: str | Path | None = None,
    template_date: str | None = None,
) -> tuple[PreTrainedModel | None, PreTrainedTokenizerBase]:
    model_name = ModelName(model_name)
    base_model_path = MODEL_REPOS[model_name]

    if lora_path is None:
        # The base model is read remotely on NDIF, so only its config is fetched here: an nnsight
        # wrapper on meta weights. Mistral-3 and Gemma-3 are multimodal classes, which nnsight's
        # LanguageModel refuses and NDIF deploys as VisionLanguageModel; token tensors without
        # images take the plain language-model path. A tokenizer-only caller builds no model.
        tokenizer = load_tokenizer(model_name, template_date)
        if omit_model:
            return None, tokenizer
        wrapper = VisionLanguageModel if model_name.is_vlm else LanguageModel
        model = wrapper(base_model_path)
        if model_name.is_vlm:
            prune_to_decoder_blocks(model)
        logger.info(f"Loaded model {model_name} from {base_model_path}")
        model.eval()
        return model, tokenizer

    if template_date is not None:
        raise NotImplementedError("template_date is only wired up for the remote base model")

    final_model_path = base_model_path

    # Determine final model path to load from
    if lora_path is not None:
        # Create merged model path based on model name and LoRA path
        lora_name = Path(lora_path).name
        merged_model_path = Path("models") / "peft" / f"{model_name.value}_merged_{lora_name}"

        # Skip merge-and-save for Gemma, Mistral, and Llama due to CUDA issues
        # Load LoRA dynamically instead
        if "gemma" not in base_model_path.lower() and "mistral" not in base_model_path.lower() and "llama" not in base_model_path.lower():
            # Check if merged model already exists
            if merged_model_path.exists() and (merged_model_path / "config.json").exists():
                logger.info(f">>> Found existing merged model at: {merged_model_path}")
                print(f">>> Found existing merged model at: {merged_model_path}")
                final_model_path = str(merged_model_path)
            else:
                logger.info(f">>> Merging LoRA adapter: {lora_path}")
                print(f">>> Merging LoRA adapter: {lora_path}")
                
                # Load base model first
                logger.info(f"Loading base model for LoRA merge: {base_model_path}")
                base_processor = None  # Initialize for type safety
                
                if "mistral" in base_model_path.lower() or "gemma" in base_model_path.lower():
                    base_model = AutoModelForImageTextToText.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        local_files_only=True,
                    )
                    base_processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left")
                    base_processor.tokenizer.chat_template = base_processor.chat_template
                    base_tokenizer = base_processor.tokenizer
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        local_files_only=True,
                    )
                    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
                
                if "qwen" in base_model_path.lower():
                    base_tokenizer.bos_token_id = base_tokenizer.eos_token_id
                    
                base_tokenizer.pad_token_id = base_tokenizer.bos_token_id
                
                # Now create PEFT model from loaded base model
                print("merging...")
                # Clear cache before merging
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                peft_model = PeftModel.from_pretrained(base_model, lora_path)  # type: ignore
                merged_model = peft_model.merge_and_unload()  # type: ignore
                
                # Clear intermediate model from memory
                del peft_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("done merging.")
                
                # Save merged model to disk
                print("saving merged model...")
                merged_model_path.mkdir(parents=True, exist_ok=True)
                
                # Move model to CPU before saving to avoid CUDA memory issues
                print("Moving model to CPU for saving...")
                # merged_model = merged_model.cpu()

                merged_model.save_pretrained(
                    merged_model_path,
                    safe_serialization=False,  # This is the key change - use PyTorch format
                    max_shard_size="4GB"
                )
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # merged_model.save_pretrained(merged_model_path)
                
                # For Mistral models, save the processor (which includes tokenizer and image processor)
                if "mistral" in base_model_path.lower() or "gemma" in base_model_path.lower(): 
                    if base_processor is not None:
                        base_processor.save_pretrained(merged_model_path)
                else:
                    base_tokenizer.save_pretrained(merged_model_path)
                print("done saving merged model")
                
                # Free memory after saving
                del merged_model
                del base_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Additional cleanup handled above
                
                logger.info(f">>> Saved merged model to: {merged_model_path}")
                print(f">>> Saved merged model to: {merged_model_path}")
                final_model_path = str(merged_model_path)
        

    # Clean up any existing GPU memory before loading new model
    if torch.cuda.is_available():
        cleanup_model_memory()
    
    # Load the final model (either base model or merged model)
    # if "mistral" in final_model_path.lower() or "mistral" in base_model_path.lower():
    #     model = AutoModelForImageTextToText.from_pretrained(
    #         final_model_path,
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         local_files_only=True,
    #     )
    #     processor = AutoProcessor.from_pretrained(final_model_path, padding_side="left")
    #     processor.tokenizer.chat_template = processor.chat_template
    #     tokenizer = processor.tokenizer
    if ("gemma" in base_model_path.lower() or "mistral" in base_model_path.lower() or "llama" in base_model_path.lower()) and lora_path is not None:
        # For Gemma/Mistral/Llama with LoRA, load dynamically (skip merge due to CUDA issues)
        # Load base model first, then apply LoRA adapter
        
        if "mistral" in base_model_path.lower() or "gemma" in base_model_path.lower():
            # Mistral and Gemma use AutoProcessor (vision-language models)
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
            # Apply LoRA adapter on top of base model
            model = PeftModel.from_pretrained(base_model, lora_path)
            
            # Load processor/tokenizer from base model
            processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left")
            tokenizer = processor.tokenizer
            
            # Copy chat_template from processor to tokenizer
            if hasattr(processor, 'chat_template'):
                tokenizer.chat_template = processor.chat_template
        else:  # llama 
            base_model = LanguageModel(base_model_path)
            """base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )"""
            # Apply LoRA adapter on top of base model
            model = PeftModel.from_pretrained(base_model, lora_path)
            
            # Load tokenizer directly (not processor)
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
    else:
        # Load model without LoRA - check if Mistral/Gemma need special handling
        if "mistral" in base_model_path.lower() or "gemma" in base_model_path.lower():
            # Mistral and Gemma use AutoModelForImageTextToText (vision-language models)
            model = AutoModelForImageTextToText.from_pretrained(
                final_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(final_model_path, padding_side="left")
            tokenizer = processor.tokenizer
            if hasattr(processor, 'chat_template'):
                tokenizer.chat_template = processor.chat_template
        else:
            # Standard causal LM (Llama, Qwen, etc.)
            model = LanguageModel(final_model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(final_model_path, padding_side="left")
    
    if "qwen" in final_model_path.lower():
        tokenizer.bos_token_id = tokenizer.eos_token_id
    
    tokenizer.pad_token_id = tokenizer.bos_token_id
    
    logger.info(f"Loaded model {model_name} from {final_model_path}")
    
    if omit_model:
        return None, tokenizer

    if model.device == torch.device("cpu"):
        logger.warning("Model is on CPU")

    model.eval()
    
    return model, tokenizer


def cleanup_model_memory():
    """Clean up GPU memory by clearing cache and garbage collection."""
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")