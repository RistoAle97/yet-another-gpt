"""A simple GPT architecture implementation."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.functional import F


@dataclass
class GPTConfig:
    """GPT configuration dataclass.

    Args:
        vocab_size: the vocabulary size (default=50304).
        block_size: the context window size (default=1024).
        d_model: the embedding dimension (default=768).
        n_heads: the number of heads in the multi-attention mechanism (default=12).
        n_layers: the number of layers (default=12).
        d_ff: dimension of the feed-forward sub-layer (default=2048).
        dropout_res: the dropout value to apply for each residual connection (default=0.1).
        dropout_mha: the dropout value to apply inside the multi-head attention (default=0.1).
        dropout_ff: the dropout value to apply inside the feed-forward network (default=0.1).
        act_ff: the activation function to employ inside the feed-forward network (default=nn.GELU).
        ln_eps: the eps value in the layer normalization (default=1e-5).
        bias: whether to add bias weights where possible (default=False).
        tie_weights: whether to tie the embeddings weights with the language modeling head (default=True).
        pad_token_id: the pad token id (default=0)
        eos_token_id: the end of sequence token id (default=1)
        the label smoothing value for the cross-entropy loss (default=0.0)
    """

    vocab_size: int = 50304
    block_size: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 2048
    dropout_res: float = 0.1
    dropout_mha: float = 0.1
    dropout_ff: float = 0.1
    act_ff: nn.Module = nn.GELU
    ln_eps: float = 1e-5
    bias: bool = True
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    label_smoothing: float = 0.0


class GPTLayer(nn.Module):
    """A single GPT layer.

    The layer is made of two sub-layers: a masked multi-head attention and feed-forward network, both preceded by layer
    normalization.
    """

    def __init__(self, config: GPTConfig):
        """Initializes a GPT layer.

        Args:
            config: a GPTConfig object.
        """
        super().__init__()

        # Masked multi-head attention
        self.mha_ln = nn.LayerNorm(config.d_model, config.ln_eps, bias=config.bias)
        self.mha = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            config.dropout_mha,
            config.bias,
            batch_first=True,
        )
        self.mha_dropout_res = nn.Dropout(config.dropout_res)

        # Feed-forward
        self.ff_ln = nn.LayerNorm(config.d_model, config.ln_eps, bias=config.bias)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, config.bias),
            config.act_ff,
            nn.Linear(config.d_ff, config.d_model, config.bias),
            nn.Dropout(config.dropout_ff),
        )
        self.ff_dropout_res = nn.Dropout(config.dropout_res)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass inside the layer.

        Args:
            x: input tensor of shape (bsz, seq_len, d_model), this should be the embeddings.
            padding_mask: padding mask of shape (bsz, seq_len) where True values indicate positions that must not be
                attended as they represent pad tokens.
            attn_mask: causal attention mask of shape (bsz, seq_len) that prevents tokens from attending on future ones.
                True values indicates that the position must not be considered.

        Returns:
            torch.Tensor: tensor of shape (bsz, seq_len, d_model) that is the layer output.
        """
        # Masked multi-head attention
        mha_out = self.mha_ln(x)
        mha_out = self.mha.forward(mha_out, mha_out, mha_out, padding_mask, attn_mask=attn_mask)[0]
        mha_out = x + self.mha_dropout_res(mha_out)

        # Feed-forward
        ff_out = self.ff_ln(mha_out)
        ff_out = self.ff(ff_out)
        layer_out = mha_out + self.ff_dropout_res(ff_out)
        return layer_out


class GPT(nn.Module):
    """Model based on the GPT2 architecture."""

    def __init__(self, config: GPTConfig) -> None:
        """Initializes a GPT model.

        Args:
            config: a GPTConfig object.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.pos_embeddings = nn.Embedding(config.block_size, config.d_model)
        self.embeddings_dropout_res = nn.Dropout(config.dropout_res)

        # Layers and final layer norm
        self.layers = nn.ModuleList([GPTLayer(config) for _ in range(config.n_layers)])
        self.last_ln = nn.LayerNorm(config.d_model, config.ln_eps, bias=config.bias)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        # Initiliaze the weights
        self.apply(self._init_bert_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_bert_weights(module: nn.Module) -> None:
        """Initialize module's weights following BERT https://arxiv.org/pdf/1810.04805.pdf.

        The weights of the nn.Linear and nn.Embedding layers are sampled by a normal distribution with mean 0.0 and std
        0.02, while the weights of nn.LayerNorm layer are set to 1.0. The bias of the nn.Linear, nn.LayerNorm layers
        and the weights related to the padding token inside the nn.Embedding are then set to 0.

        Args:
            module: the PyTorch nn.Module to initialize.
        """
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Performs a forward pass.

        Args:
            x: inputs ids of shape (bsz, seq_len).
            padding_mask: boolean mask of shape (bsz, seq_len) where True indicates padding.
            targets: expected values of shape (bsz, seq_len) used during training (default=None).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: a tuple containing the logits coming from the language modeling
                head and the loss if the targets tensor is passed.
        """
        _, seq_len = x.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}."
            )

        # Embeddings
        pos = torch.arange(seq_len, device=x.device)  # (seq_len)
        x = self.token_embeddings(x) + self.pos_embeddings(pos)  # (bsz, seq_len, d_model)
        x = self.embeddings_dropout_res(x)

        # Forward pass through the layers
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        for layer in self.layers:
            x = layer(x, padding_mask, attn_mask)  # (bsz, seq_len, d_model)

        # Language modeling head
        logits = self.lm_head(x)  # (bsz, seq_len, vocab_size)

        # Compute loss if targets tensor is not None
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temp: float = 1.0) -> torch.Tensor:
        """Generate tokens at inference time using greedy decoding.

        Args:
            input_ids: tokenized source sentences of shape (bsz, seq_len).
            max_new_tokens: the maximum number of tokens that can be  generated (default=50).
            temp: the temperature value (default=1.0).

        Returns:
            torch.Tensor: the source tokens concatenated with the newly generated ones.

        Raises:
            ValueError: if ``max_new_tokens`` or ``temp`` are not higher than 0.
        """
        self.eval()

        # Checks before starting
        if max_new_tokens <= 0:
            raise ValueError("The parameter 'max_new_tokens' must be higher than 0.")

        if temp <= 0:
            raise ValueError("The 'temperature' must be higher than 0.")

        # Some useful parameters
        device = input_ids.device
        bsz, seq_len = input_ids.size()

        # Keep track of unfinished sentences (those that did not generate an eos token)
        unfinished_sentences = input_ids.new(bsz).fill_(1)
        eos_token_id_tensor = torch.tensor([self.config.eos_token_id]).unsqueeze(1).to(device)

        # Generate tokens in an autoregressive way
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            input_ids_block = (
                input_ids if seq_len <= self.config.block_size else input_ids[:, -self.config.block_size :]
            )

            # Create the padding mask
            padding_mask = input_ids_block.eq(self.config.pad_token_id).to(device)

            # Call the model and scale the logits with the temperature
            logits, _ = self(input_ids_block, padding_mask)
            logits = logits[:, -1] / temp

            # Compute the new tokens and concatenate them to the previously generated ones
            p_logits = F.log_softmax(logits[:, -1], -1)
            new_tokens = p_logits.argmax(-1)
            new_tokens = new_tokens * unfinished_sentences + self.config.pad_token_id * (1 - unfinished_sentences)
            input_ids = torch.cat([input_ids, new_tokens[:, None]], dim=-1)

            # Update tensor that tracks unfinished sentences
            unfinished_sentences = unfinished_sentences.mul(new_tokens.ne(eos_token_id_tensor).prod(dim=0))

            # Terminate if an eos token was generated for all sentences
            if unfinished_sentences.max() == 0:
                break

        return input_ids
