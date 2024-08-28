#!/usr/bin/env bash
pip3 install peewee pytorch-lightning numpy chess
curl https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz --output 2021-07-31-lichess-evaluations-37MM.db.gz
gzip -d "2021-07-31-lichess-evaluations-37MM.db.gz"