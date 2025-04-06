#!/bin/bash

# Define URLs and output filenames in order
urls=(
  "https://jax-ml.github.io/scaling-book/roofline"
  "https://jax-ml.github.io/scaling-book/tpus"
  "https://jax-ml.github.io/scaling-book/sharding"
  "https://jax-ml.github.io/scaling-book/transformers"
  "https://jax-ml.github.io/scaling-book/training"
  "https://jax-ml.github.io/scaling-book/applied-training"
  "https://jax-ml.github.io/scaling-book/inference"
  "https://jax-ml.github.io/scaling-book/applied-inference"
  "https://jax-ml.github.io/scaling-book/profiling"
  "https://jax-ml.github.io/scaling-book/jax-stuff"
  "https://jax-ml.github.io/scaling-book/conclusion"
)

files=(
  "roofline.pdf"
  "tpus.pdf"
  "sharding.pdf"
  "transformers.pdf"
  "training.pdf"
  "applied-training.pdf"
  "inference.pdf"
  "applied-inference.pdf"
  "profiling.pdf"
  "jax-stuff.pdf"
  "conclusion.pdf"
)

# Launch headless Chrome for each URL in parallel
for i in "${!urls[@]}"; do
  /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --headless --disable-gpu --print-to-pdf="${files[i]}" "${urls[i]}" &
done

# Wait for all background processes to finish
wait

# Merge the generated PDFs in order
pdfunite "${files[@]}" combined.pdf
