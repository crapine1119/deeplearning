if __name__ == "__main__":

    import torch
    from transformers import AutoModelForMaskedLM
    from transformers import AutoTokenizer

    tokenzier = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    bert = AutoModelForMaskedLM.from_pretrained("kykim/bert-kor-base")

    input_str = "안녕하세요"
    x = torch.LongTensor([tokenzier.encode("안녕하세요")])
    y = bert(x)
    y.logits.shape
