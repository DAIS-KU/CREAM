from .incremental_train_ranking import evaluate_cosine as inc_evaluate_cosine
from .incremental_train_ranking import evaluate_term as inc_evaluate_term
from .incremental_train_ranking import train as inc_train

from .l2r_train_ranking import evaluate as l2r_evaluate
from .l2r_train_ranking import train as l2r_train
from .proposal_train_ranking import evaluate as proposal_evaluate
from .proposal_train_ranking import eval_rankings as proposal_evaluate_rankings
from .proposal_train_ranking import train as proposal_train
from .proposal_wo_cluster_train_ranking_rerank import (
    evaluate as evaluate_wo_cluster_rerank,
)
from .proposal_wo_cluster_train_ranking_rerank import train as train_wo_cluster_rerank
from .proposal_wo_cluster_train_ranking import evaluate as evaluate_wo_cluster
from .proposal_wo_cluster_train_ranking import train as train_wo_cluster
from .proposal_wo_term_train_ranking import evaluate as evaluate_wo_term
from .proposal_wo_term_train_ranking import train as train_wo_term
from .proposal_wo_term_train_ranking_rerank import evaluate as evaluate_wo_term_rerank
from .proposal_wo_term_train_ranking_rerank import train as train_wo_term_rerank
from .warmingup import evaluate_cosine as warmingup_evaluate_cosine
from .warmingup import evaluate_term as warmingup_evaluate_term
from .warmingup import train as waringup_train

from .domain_dependency import train as dodp_train
from .domain_forget import train as df_train


from .proposal_train_ranking_qq_low import train as qq_low_train
from .proposal_train_ranking_qq_low import evaluate as qq_low_evaluate
from .proposal_train_ranking_qq_low2 import train as qq_low_train2


from .generate_baseline_answer import create_cos_ans_file
