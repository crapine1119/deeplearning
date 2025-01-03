from random import shuffle
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, softmax
from tqdm import tqdm

with open(
    "/Users/songhak/Downloads/(텍스트문서 txt) 키다리 아저씨 (우리말 옮김)(2차 편집최종)(블로그업로드용 2018년 최종) 180120.txt",
    encoding="cp949",
) as f:
    data = f.read()
    data = data.replace("\n\n", "\n")
    vocab = sorted(set(data))

s2i = {v: enum for enum, v in enumerate(vocab)}
i2s = {v: k for k, v in s2i.items()}


def encoder(text: str) -> list[int]:
    return [s2i[c] for c in text]


def decoder(tokens: list[int]) -> list[str]:
    return [i2s[t] for t in tokens]


print("Data:")
print(data[100:200])
print("Token:")
print(encoder(data[100:200]))


index_split = int(len(data) * 0.9)
train_data = encoder(data[:index_split])
valid_data = encoder(data[index_split:])


##
class Projection(nn.Module):
    def __init__(self, channel_size: int):
        super().__init__()
        self._proj = nn.Linear(channel_size, channel_size * 3)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3  # B, T, C
        x = self._proj(x)
        q, k, v = torch.tensor_split(x, 3, -1)
        return q, k, v


class Attention(nn.Module):
    def __init__(self, channel_size: int, num_head: int = 4):
        super().__init__()
        self._proj_layer = Projection(channel_size)
        self._num_head = num_head

    def forward(self, x: torch.Tensor):
        q, k, v = self._proj_layer(x)  # B, T, C
        b, t, c = q.shape
        q = q.view(b, t, c // self._num_head, self._num_head)
        k = k.view(b, t, c // self._num_head, self._num_head)
        v = v.view(b, t, c // self._num_head, self._num_head)

        q = q.permute(0, 3, 1, 2)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        weight = q @ k.transpose(-2, -1)
        d = q.shape[-1]
        scale = d**0.5
        mh_attn = weight / scale
        mask = torch.ones_like(mh_attn).tril()
        mh_attn.masked_fill_(mask == 0, float("-inf"))
        mh_attn = torch.softmax(mh_attn, dim=-1)
        mh_attn = mh_attn @ v

        attn = mh_attn.permute(0, 2, 3, 1).flatten(-2, -1)
        return attn


class DecoderLayer(nn.Module):
    def __init__(self, channel_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._channel_size = channel_size
        self._layer_norm1 = nn.LayerNorm(channel_size)
        self._layer_norm2 = nn.LayerNorm(channel_size)
        self._attn_layer = Attention(channel_size)
        self._ffw_layer = self._make_ffw_layer(channel_size)

    def _make_ffw_layer(self, channel_size: int, expansion_size: int = 4):
        return nn.Sequential(
            *[
                nn.Linear(channel_size, channel_size * expansion_size),
                nn.ReLU(),
                nn.Linear(channel_size * expansion_size, channel_size),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._layer_norm1(self._attn_layer(x) + x)
        out = self._layer_norm2(self._ffw_layer(out) + out)
        return out


class Decoder(nn.Module):
    def __init__(self, channel_size: int, num_layer: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._channel_size = channel_size
        self._layer_num = num_layer
        self._layers = self._make_decoder_layers(channel_size, num_layer)

    def _make_decoder_layers(self, channel_size: int, layer_num: int):
        return nn.Sequential(*[DecoderLayer(channel_size) for _ in range(layer_num)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, n_head: int = 8, position_len: int = 128):
        super().__init__()
        self._embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self._position_embedding = nn.Embedding(num_embeddings=position_len, embedding_dim=embedding_dim)
        # self._decoder = self._get_predifined_decoder(embedding_dim, n_head)
        self._decoder = Decoder(channel_size=embedding_dim, num_layer=4)
        self._fc_layer = nn.Linear(embedding_dim, vocab_size)

    def _get_predifined_decoder(self, embedding_dim, n_head):
        # trick bcz TransformerEncoder is gpt style decoder
        decoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=n_head)
        return nn.TransformerEncoder(decoder_layer, num_layers=n_head)

    def forward(self, x: torch.LongTensor, y: torch.LongTensor) -> tuple[Any, Tensor]:
        assert len(x.shape) == 2
        b, t = x.shape
        pos = torch.arange(t).to(x.device)
        embedding = self._embedding_layer(x)
        output = self._decoder(embedding + self._position_embedding(pos))
        logits = self._fc_layer(output)

        loss = cross_entropy(logits.permute(0, 2, 1), y)
        return logits, loss

    def generate(self, x: torch.LongTensor, max_token_len: int = 64):
        for _ in range(max_token_len):
            logits, _ = self(x, x)
            prob = softmax(logits[:, -1])  # B, C
            x_next = torch.multinomial(prob, num_samples=1)
            x = torch.cat([x, x_next], dim=-1)
            print("".join(decoder(x.squeeze().tolist())), flush=True, end="")
            # time.sleep(0.1)
        return "".join(decoder(x.squeeze().tolist()))


def main():
    epoch = 15
    batch_size = 16
    sequence_len = 16 + 1
    train_dataset = [
        torch.LongTensor(train_data[i : i + sequence_len]) for i in range(0, len(train_data), sequence_len)
    ]
    train_dataset = train_dataset[:-1]
    valid_dataset = [
        torch.LongTensor(valid_data[i : i + sequence_len]) for i in range(0, len(valid_data), sequence_len)
    ]
    valid_dataset = valid_dataset[:-1]
    shuffle(train_dataset)

    def get_dataloader(dataset: list[torch.LongTensor], batch_size: int) -> list[tuple[Tensor, Tensor]]:
        dataloader = [torch.stack(dataset[i : i + batch_size]) for i in range(0, len(dataset), batch_size)][:-1]
        return [(x[:, :-1], x[:, 1:]) for x in dataloader]

    train_dataloader = get_dataloader(train_dataset, batch_size)
    valid_dataloader = get_dataloader(valid_dataset, 1)

    print("Input size: ")
    print(train_dataloader[0][0].shape)

    model = GPT(vocab_size=len(vocab))
    # test
    model.eval()
    logits = model(*valid_dataloader[0])
    model.to(torch.device("mps"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for e in range(epoch):
        for x, y in tqdm(train_dataloader, postfix=f"epoch: {e}"):
            optimizer.zero_grad()
            _, loss = model(x.to("mps"), y.to("mps"))
            loss.backward()
            optimizer.step()
            print(loss.item())

    model.eval()
    model.generate(torch.LongTensor([[s2i["학"]]]).to("mps"))
    print("finish")


if __name__ == "__main__":
    main()
