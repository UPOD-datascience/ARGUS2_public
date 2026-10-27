#!/bin/bash

#read in from .env
export $(grep -v '^#' .env | xargs)

rsconnect deploy manifest manifest.json \
    --server https://rsc.ds.umcutrecht.nl \
    --api-key $PERSONAL_API_KEY \
    --new \

