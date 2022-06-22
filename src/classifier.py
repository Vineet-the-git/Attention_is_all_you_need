import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext.legacy import data, datasets, vocab
import numpy as np
from argparse import ArgumentParser
import tqdm, random, math, gzip
from text_classification import Transformer

LOG2E = math.log2(math.e)
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLS = 2

def main(args):
    # load the IMDB data
    if args.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)

    TEXT.build_vocab(train, max_size=args.vocab_size - 2) # - 2 to make space for <unk> and <pad>
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=args.batch_size, device='cuda')
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of {"test" if args.final else "validation"} examples {len(test_iter)}')

    if args.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = args.max_length

    # create the model
    model = Transformer(emb = args.embedding_size, heads=args.num_heads, depth=args.depth, seq_length=mx, num_tokens=args.vocab_size, num_classes=NUM_CLS)
    if torch.cuda.is_available():
        model.cuda()
    
    opt = torch.optim.Adam(lr=args.lr, params=model.parameters())
    
    # training loop
    seen = 0
    for epoch in range(args.num_epochs):
        print(f'\n epoch: {epoch}')
        model.train(True)

        for batch in tqdm.tqdm(train_iter):
            opt.zero_grad()
            input = batch.text[0]
            label = batch.label-1

            if input.size(1) > mx:
                input = input[:,:mx]
            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()

            opt.step()

            seen += input.size(0)

        with torch.no_grad():
            model.train(False)
            tot, cor = 0.0, 0.0

            for batch in test_iter:
                input = batch.text[0]
                label = batch.label-1

                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input).argmax(dim = 1)

                tot += float(input.size(0))
                cor += float((label==out).sum().item())

            acc = cor/tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=123, type=int)

    args = parser.parse_args()

    print('arguments: ', args)

    main(args)