#!/usr/bin/env bash

# Experiment 1. BK-expert & DCE-expert
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_critique.sh
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_critique.sh

# Experiment 1. BK-normal & DCE-normal
# Experiment 3. without clarification
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_normal.sh
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_normal.sh

# Experiment 2. BK-random & BK-NN
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_cc_random.sh
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_cc_neighbour.sh

# Experiment 2. DCE-Tree
# Experiment 3. non-personalized clarification & personalized clarification
sbatch --time=10:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_cc_tree.sh