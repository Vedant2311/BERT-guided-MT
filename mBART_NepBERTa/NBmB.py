import torch
from torch import nn
import torch.nn.functional as F

class NBmB(nn.Module):
    def __init__(self, mBart_model, nepBerta_model, mBart_tokenizer):
        super().__init__()
        self.mBart_model = mBart_model
        self.mBart_embed = mBart_model.get_input_embeddings()
        self.nepBerta_model = nepBerta_model
        self.fc = nn.Linear(768, 1024)
        self.mBart_tokenizer = mBart_tokenizer

    def forward(self, input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len):        
        # NepBERTa
        nepBerta_output = self.nepBerta_model(input_ids=input_ids, attention_mask=encoder_mask)
        nepBerta_encoding = nepBerta_output.hidden_states[-1]
        nepBerta_encoding = self.fc(nepBerta_encoding)
        nepBerta_encoding[:, 0, :] = 0

        mBart_encoding = self.mBart_embed(mbart_input_ids)

        # Fuse by adding the two matrices
        fused_encoding = torch.add(nepBerta_encoding, mBart_encoding)

        fused_output = self.mBart_model(
            inputs_embeds=fused_encoding,
            attention_mask=mbart_input_mask,
            labels = labels,
            use_cache=False,
        )
        return fused_output
    
    # Using beam search for inference
    def generate(self, input_ids, mbart_input_ids):
        # NepBERTa
        nepBerta_output = self.nepBerta_model(input_ids=input_ids)
        nepBerta_encoding = nepBerta_output.hidden_states[-1]
        nepBerta_encoding = self.fc(nepBerta_encoding)

        # zero out the first token
        nepBerta_encoding[:, 0, :] = 0

        mBart_encoding = self.mBart_embed(mbart_input_ids)
        
        # make the sizes same by padding zeros
        if nepBerta_encoding.size(1) < mBart_encoding.size(1):
            padding = torch.zeros(nepBerta_encoding.size(0), mBart_encoding.size(1) - nepBerta_encoding.size(1), nepBerta_encoding.size(2)).to(nepBerta_encoding.device)
            nepBerta_encoding = torch.cat((nepBerta_encoding, padding), dim=1).to(input_ids.device)
        elif nepBerta_encoding.size(1) > mBart_encoding.size(1):
            padding = torch.zeros(mBart_encoding.size(0), nepBerta_encoding.size(1) - mBart_encoding.size(1), mBart_encoding.size(2)).to(mbart_input_ids.device)
            mBart_encoding = torch.cat((mBart_encoding, padding), dim=1).to(mbart_input_ids.device)


        # Fuse by adding the two matrices
        fused_encoding = torch.add(nepBerta_encoding, mBart_encoding)

        inputs = {"inputs_embeds": fused_encoding}

        generated_ids = self.mBart_model.generate(
            **inputs,
            use_cache=True,
            decoder_start_token_id=self.mBart_tokenizer.lang_code_to_id["en_XX"],
            num_beams=5,
            max_length=512,
            early_stopping=True,
        )

        return self.mBart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)