# SSPG

Code for the subword segmental pointer generator (SSPG) proposed in the paper [Triples-to-isiXhosa (T2X): Addressing the Challenges of Low-Resource Agglutinative Data-to-Text Generation](https://aclanthology.org/2024.lrec-main.1464.pdf), Francois Meyer and Jan Buys, LREC-COLING 2024.

SSPG is implemented as a model in fairseq. The code in this repo can be used to train new SSPG models for data-to-text. Trained SSPG models can be used to generate text from data using either unmixed or dynamic decoding. The SSPG models trained for our paper (for isiXhosa and Finnish data-to-text) are publicly available:
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
    $DATA_DIR --task subword_segmental_data2text --source-lang data --target-lang text \
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

### Citation

```bibtex
@inproceedings{meyer-buys-2024-triples,
    title = "Triples-to-isi{X}hosa ({T}2{X}): Addressing the Challenges of Low-Resource Agglutinative Data-to-Text Generation",
    author = "Meyer, Francois  and
      Buys, Jan",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1464/",
    pages = "16841--16854",
    abstract = "Most data-to-text datasets are for English, so the difficulties of modelling data-to-text for low-resource languages are largely unexplored. In this paper we tackle data-to-text for isiXhosa, which is low-resource and agglutinative. We introduce Triples-to-isiXhosa (T2X), a new dataset based on a subset of WebNLG, which presents a new linguistic context that shifts modelling demands to subword-driven techniques. We also develop an evaluation framework for T2X that measures how accurately generated text describes the data. This enables future users of T2X to go beyond surface-level metrics in evaluation. On the modelling side we explore two classes of methods - dedicated data-to-text models trained from scratch and pretrained language models (PLMs). We propose a new dedicated architecture aimed at agglutinative data-to-text, the Subword Segmental Pointer Generator (SSPG). It jointly learns to segment words and copy entities, and outperforms existing dedicated models for 2 agglutinative languages (isiXhosa and Finnish). We investigate pretrained solutions for T2X, which reveals that standard PLMs come up short. Fine-tuning machine translation models emerges as the best method overall. These findings underscore the distinct challenge presented by T2X: neither well-established data-to-text architectures nor customary pretrained methodologies prove optimal. We conclude with a qualitative analysis of generation errors and an ablation study."
}
```
