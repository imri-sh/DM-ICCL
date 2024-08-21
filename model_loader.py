from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # The default, "full", dtype


def set_dtype_fp8():
    """Sets the default floating point of the models to 8 bytes (1/4th the "full" size)"""
    dtype = torch.float8


def set_dtype_fp16():
    """Sets the default floating point of the models to 16 bytes (half the "full" size)"""
    dtype = torch.float16


def set_dtype_fp32():
    """Sets the default floating point of the models to 32 bytes (the "full" size)"""
    dtype = torch.float16


def get_phi2():
    """ google's phi-2. 2.7B params."""
    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer, "phi-2"


def get_phi3():
    """ google's phi-3. 3.8B params."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer, "Phi-3"


def get_phi3_5():
    """ google's phi-3. 3.8B params."""
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 trust_remote_code=True, ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer, "Phi-3.5"


def get_flan_T5_base():
    """ Microsoft's FLAN T5 base. 250M params."""
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer, "flan-t5-base"


def get_flan_T5_large():
    """ Microsoft's FLAN T5 large. 780M params."""
    model_name = "google/flan-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer, "flan-t5-large"


def get_flan_T5_xl():
    """ Microsoft's FLAN T5 xl. 3B params."""
    model_name = "google/flan-t5-xl"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer, "flan-t5-xl"
