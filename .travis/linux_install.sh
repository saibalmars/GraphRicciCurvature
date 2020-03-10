#!/usr/bin/env bash
set -ex

# create new empty venv
virtualenv -p python ~/venv
source ~/venv/bin/activate

set +ex