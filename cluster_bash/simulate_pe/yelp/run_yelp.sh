#!/usr/bin/env bash

# For abs diff baseline
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_critique.sh
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_critique.sh

# For abs diff normal user - surrogate critiques
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_normal.sh
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_normal.sh

# For BK clarification methods, baselines
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_cc_random.sh
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL BK_cc_neighbour.sh

# For DCE clarification methods
sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_cc_tree.sh

# For DCE clarification methods, is actually not a valid baseline ...
#sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_cc_neighbour.sh
#sbatch --time=09:00:00 --mem=32G --cpus-per-task=4 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_cc_random.sh
