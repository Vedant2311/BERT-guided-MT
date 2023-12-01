import torch
from torch import nn

class NBmT5(nn.Module):
    def __init__(self, mT5_model, nepBerta_model, mT5_tokenizer):
        super().__init__()
        self.mT5_model = mT5_model
        self.mT5_embed = mT5_model.get_input_embeddings()
        self.nepBerta_model = nepBerta_model
        self.mT5_tokenizer = mT5_tokenizer

    def forward(self, input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len):        
        # NepBERTa
        nepBerta_output = self.nepBerta_model(input_ids=input_ids, attention_mask=encoder_mask)
        nepBerta_encoding = nepBerta_output.hidden_states[-1]

        mT5_encoding = self.mT5_embed(mT5_input_ids)

        # Fuse by adding the two matrices
        fused_encoding = torch.add(nepBerta_encoding, mT5_encoding)

        fused_output = self.mT5_model(
            inputs_embeds=fused_encoding,
            attention_mask=mT5_input_mask,
            labels = labels,
            use_cache=False
        )
        return fused_output
    
    # Using beam search for inference
    def generate(self, input_ids, mT5_input_ids):
        # NepBERTa
        nepBerta_output = self.nepBerta_model(input_ids=input_ids)
        nepBerta_encoding = nepBerta_output.hidden_states[-1]

        mT5_encoding = self.mT5_embed(mT5_input_ids)
        # make the sizes same by padding zeros
        if nepBerta_encoding.size(1) < mT5_encoding.size(1):
            padding = torch.zeros(nepBerta_encoding.size(0), mT5_encoding.size(1) - nepBerta_encoding.size(1), nepBerta_encoding.size(2)).to(nepBerta_encoding.device)
            nepBerta_encoding = torch.cat((nepBerta_encoding, padding), dim=1).to(input_ids.device)
        elif nepBerta_encoding.size(1) > mT5_encoding.size(1):
            padding = torch.zeros(mT5_encoding.size(0), nepBerta_encoding.size(1) - mT5_encoding.size(1), mT5_encoding.size(2)).to(mT5_input_ids.device)
            mT5_encoding = torch.cat((mT5_encoding, padding), dim=1).to(mT5_input_ids.device)

        # Fuse by adding the two matrices
        fused_encoding = torch.add(nepBerta_encoding, mT5_encoding)

        inputs = {"inputs_embeds": fused_encoding}

        generated_ids = self.mT5_model.generate(
            **inputs,
            use_cache=True,
            num_beams=5,
            max_length=512,
            early_stopping=True
        )

        return self.mT5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)