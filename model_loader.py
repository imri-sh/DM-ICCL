from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

dtype = torch.float32  # The default, "full", dtype

def set_dtype(fp_type):
    global dtype
    if fp_type == 'fp32':
        dtype = torch.float32
    elif fp_type == 'fp16':
        dtype = torch.float16
    elif fp_type == 'fp8':
        dtype = torch.float8
    else:
        raise Exception(f'{fp_type} currently not supported.')

class ModelLoader:
    @staticmethod
    def get_model_and_tokenizer(model_name, device='cpu'):
        if model_name == 'phi2':
            """ google's phi-2. 2.7B params."""
            model_name = "microsoft/phi-2"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            return model, tokenizer, "phi2"

        elif model_name == 'phi3':
            """ google's phi-3. 3.8B params."""
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            return model, tokenizer, "phi3"

        elif model_name == 'phi3_5':
            """ google's phi-3. 3.8B params."""
            model_name = "microsoft/Phi-3.5-mini-instruct"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                         trust_remote_code=True, ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            return model, tokenizer, "phi3_5"

        elif model_name == 'phi3_5_MoE':
            model_name = "microsoft/Phi-3.5-MoE-instruct"
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            return model, tokenizer, "phi3_5_MoE"

        elif model_name == 'flan_t5_base':
            """ Microsoft's FLAN T5 base. 250M params."""
            model_name = "google/flan-t5-base"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            return model, tokenizer, "flan_t5_base"

        elif model_name == 'flan_t5_large':
            """ Microsoft's FLAN T5 large. 780M params."""
            model_name = "google/flan-t5-large"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            return model, tokenizer, "flan_t5_large"

        elif model_name == 'flan_t5_xl':
            """ Microsoft's FLAN T5 xl. 3B params."""
            model_name = "google/flan-t5-xl"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            return model, tokenizer, "flan_t5_xl"

        elif model_name == 'llama3_8b_instruct':
            """Meta's Llama 3 8B instruct. 8B params."""
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
                device
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer, "llama3_8b_instruct"

        elif model_name == 'gemma2_9b_instruct':
            """Google's Gemma 2 9B instruct. 9B params."""
            model_name = "google/gemma-2-9b-it"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
                device
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer, "gemma2_9b_instruct"
        elif model_name == "gemma2_9b":
            model_name = "google/gemma-2-9b"
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer, "gemma2_9b"
        elif model_name == "llama_3_8b":
            model_name = "meta-llama/Meta-Llama-3-8B"
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
                device
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer, "llama_3_8b"

        else:
            raise Exception(f'{model_name} currently not supported.')
