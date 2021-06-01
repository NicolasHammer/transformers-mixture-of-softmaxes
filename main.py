import argparse
import math
import time

import numpy as np
import torch
from torch import nn

import Corpus as c
from model import make_transformer, make_mos_transformer

parser = argparse.ArgumentParser(description='Transformer-MOS')
parser.add_argument('--data', type=str, default='./data/penntreebank', help='location of corpus')
parser.add_argument('--mos', default=False6, action='store_true', help='use mixture of softmax decoder')
parser.add_argument('--mixtures', type=int, default=10, help='num mixtures of softmax')
parser.add_argument('--dmodel', type=int, default=300, help='dimension of model')
parser.add_argument('--layers', type=int, default=4, help='number of transformer encoder layers')
parser.add_argument('--ffhidden', type=int, default=300, help='number of feed forward hidden units')
parser.add_argument('--dropout', type=float, default=.2, help='dropout rate')
parser.add_argument('--nhead', type=int, default=4, help='number of attention heads')
parser.add_argument('--seed', type=int, default=26, help='seed')
parser.add_argument('--cuda', default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--lr', type=float, default=7, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='num epochs')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')

EVAL_BATCH_SIZE = 10
MODEL_SAVE_DIR = './model'

model = None
ntokens = 0
criterion = nn.NLLLoss()


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def train_epoch(train_data, epoch, args, lr):
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_mask(args.bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        if data.size(0) != args.bptt:
            src_mask = model.generate_mask(data.size(0)).to(device)
        model.zero_grad()
        output = model(data, src_mask)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train(train_data, val_data, args):
    best_val_loss = float("inf")
    lr = args.lr
    for epoch in range(1, args.epochs + 1):
        train_epoch(train_data, epoch, args, lr)
        val_loss = evaluate(val_data, args)
        print('=' * 89)
        print('| End of epoch {} | val loss {:5.2f} | val ppl {:8.2f}'.format(epoch, val_loss, math.exp(val_loss)))
        print('=' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_weights.pth')
        else:
            lr = lr / 1.75


def evaluate(data_source, args):
    model.eval()
    total_loss = 0.0
    src_mask = model.generate_mask(args.bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != args.bptt:
                src_mask = model.generate_mask(data.size(0)).to(device)
            output = model(data, src_mask)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available() is False:
        raise Exception("CUDA is not available for use try running without --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    print("USING {}".format(device))

    corpus = c.Corpus(args.data, device)
    ntokens = len(corpus.dictionary)

    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, EVAL_BATCH_SIZE)
    test_data = batchify(corpus.test, EVAL_BATCH_SIZE)

    make = make_mos_transformer if args.mos else make_transformer
    if args.mos:
        model = make_mos_transformer(n_experts=args.mixtures, n_tokens=ntokens, dim_model=args.dmodel,
                                     n_heads=args.nhead, n_layers=args.layers,
                                     n_ff_hid=args.ffhidden, dropout=args.dropout)
        print("Using MOS")
    else:
        model = make_transformer(n_tokens=ntokens, dim_model=args.dmodel, n_heads=args.nhead, n_layers=args.layers,
                                 n_ff_hid=args.ffhidden, dropout=args.dropout)
        print("Using non-mos")

    total_params = sum(x.data.nelement() for x in model.parameters())
    print("total number of params: {}".format(total_params))

    model.to(device)

    LR = args.lr

    try:
        print('-' * 100)
        print("Starting training...")
        train(train_data, val_data, args)
    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training...')

    test_loss = evaluate(test_data, args)
    print('=' * 100)
    print('|test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 100)
