#!/usr/bin/env bash
pip3 install peewee pytorch-lightning
wget https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz
gzip -d "2021-07-31-lichess-evaluations-37MM.db.gz"
rm "2021-07-31-lichess-evaluations-37MM.db.gz"