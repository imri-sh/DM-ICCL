from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


def get_phi2():
    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_base():
    model_name = "microsoft/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_large():
    model_name = "microsoft/flan-t5-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer


def get_flan_T5_xl():
    model_name = "microsoft/flan-t5-xl"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    return model, tokenizer
