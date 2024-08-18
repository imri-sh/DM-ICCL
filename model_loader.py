from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def get_phi2():
    """ google's phi-2. 2.7B params."""
    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_phi3():
    """ google's phi-3. 3.8B params."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_base():
    """ Microsoft's FLAN T5 base. 250M params."""
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_large():
    """ Microsoft's FLAN T5 large. 780M params."""
    model_name = "google/flan-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_xl():
    """ Microsoft's FLAN T5 xl. 3B params."""
    model_name = "google/flan-t5-xl"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer
