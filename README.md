# NeuroStr  neural network based strings

## Description

Doing strings command to a memory dump or binary file generates many amount of crap strings.
This tool use a trained neural network, and shows the strings, also show some crap but reduces the crap in 85%
I haven't identified false negatives in several tests, but thre are some false positives (crap strings)

```bash
~/s/neurostr ❯❯❯ target/release/neurostr exec model.ai /bin/ls | wc -l                                  main ✱
510
~/s/neurostr ❯❯❯ strings /bin/ls | wc -l                                                                main ✱
1608
```

## Usage

```bash
make
target/release/neural exec model.ai memory.dump
```

## Train a model

you need a good.txt with all type of good information, and crap.bin with some crap, it aslso uses random generator to generate random datasets.

```bash
make
target/release/neural train good.txt mymodel.ai
```

is necesary a file named crap.bin but can be empty, this can be used to reduce the false positives you find.


## The model model.ai

on the good.txt i used:
- password wordlist
- urls
- domains
- csv's
- ip's
- other information


