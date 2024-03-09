# SSPG

Code for the subword segmental pointer generator (SSPG) proposed in the paper *Triples-to-isiXhosa (T2X): Addressing the Challenges of Low-Resource Agglutinative Data-to-Text Generation*, Francois Meyer and Jan Buys, LREC-COLING 2024.

SSPG is implemented as a model in fairseq. The code in this repo can be used to train new SSPG models and to generate data-to-text with trained SSPG models using unmixed or dynamic decoding. The SSPG models trained for our paper (for isiXhosa and Finnish data-to-text) are publicly available:
* [SSPG for isiXhosa T2X data-to-text](https://drive.google.com/file/d/1JQEN_Fu0JfBqLgI5MUNQKbAjcuPbYAM6/view?usp=sharing)
* [SSPG for Finnish Hockey data-to-text](https://drive.google.com/file/d/1q52vJfj8F6iAfYawjgxDo1W0JJnxUg6Z/view?usp=sharing)

## Dependencies
* python 3
* [fairseq](https://github.com/pytorch/fairseq) (commit: 806855bf660ea748ed7ffb42fe8dcc881ca3aca0)
* pytorch 1.0.1.post2
* cuda 11.4
* nltk

## Usage
Merge the sspg files with fairseq.

```shell
git clone https://github.com/pytorch/fairseq.git
git clone https://github.com/francois-meyer/sspg

# change to 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 branch
cd fairseq
git checkout 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 

# copy files from ssmt to fairseq
cp -r ../sspg/fairseq ./ 
cp -r ../sspg/fairseq_cli ./  
```

## Instructions

1. Segment {train/valid/test}.data with BPE, leave {train/valid/test}.text unsegmented.

2. Preprocess the data files.

```shell
python fairseq/fairseq_cli/preprocess.py --dataset-impl=raw \
    --source-lang data --target-lang text \
    --trainpref $DATA_DIR/train --validpref $DATA_DIR/valid --testpref $DATA_DIR/test \
    --destdir $DATA_DIR/pre
```

3. Train SSPG model. Setting the `--decoder-copy` argument equips the subword segmental sequence-to-sequence model with a copy mechanism (pointer generator). 

```shell
python fairseq/fairseq_cli/train.py --dataset-impl=raw \
    $DATA_DIR --task subword_segmental_data2text --source-lang data --target-lang text\
    --max-epoch 50 --optimizer adam --lr 0.001 --lr-scheduler inverse_sqrt \
    --arch ssd2t --criterion subword_segmental_cross_entropy \
    --encoder-bidirectional --decoder-attention True --decoder-copy \
    --max-seg-len 5 --lexicon-max-size 1000 --batch-size 4 --dropout 0.5 \
    --encoder-embed-dim 128 --encoder-hidden-size 128 --encoder-layers 1 --decoder-layers 1 \
    --decoder-embed-dim 128 --decoder-hidden-size 128 --decoder-out-embed-dim 128 \
    --vocabs-path $OUT_DIR --no-epoch-checkpoints --save-dir $OUT_DIR &>> $OUT_DIR/log

```

4. Run generate_ssd2t.py to generate text based on data.

```shell
python fairseq/fairseq_cli/generate_ssd2t.py \
    $DATA_DIR --dataset-impl=raw --task subword_segmental_data2text \
    --source-lang data --target-lang text --max-len-b 500 \
    --path $OUT_DIR/checkpoint_best.pt \
    --batch-size 64 --beam 5 --normalize-type seg-seg --decoding separate \
    --results-path $RESULTS_DIR --vocabs-path $OUT_DIR &>> $RESULTS_DIR/log

```

