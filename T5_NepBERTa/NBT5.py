from torch import nn

class NBT5(nn.Module):
    def __init__(self, t5_model, nepBerta_model, device="cuda"):
        super().__init__()
        self.t5_model = t5_model.to(device)
        self.nepBerta_model = nepBerta_model.to(device)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]

        # nepBerta_model output
        nepBerta_output = self.nepBerta_model(input_ids)
        nepBerta_logits = nepBerta_output.logits

        # mBart_model output
        mbart_output = self.t5_model(input_embeds=nepBerta_logits, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)

        return mbart_output