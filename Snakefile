"""Snakemake workflow to train and predict Isopleth and Isopleth + meta-learning model"""

from pathlib import Path


configfile: "config.yaml"


BASE = Path(workflow.basedir).resolve()


def local_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else BASE / path)


INPUTS = {name: local_path(path) for name, path in config["inputs"].items()}
PREPROCESSING_CONFIG = config.get("preprocessing", {})
PREPROCESSING = {
    name: local_path(path)
    for name, path in PREPROCESSING_CONFIG.items()
    if name not in {"enabled", "output_dir"}
}
PREPROCESSING_ENABLED = PREPROCESSING_CONFIG.get("enabled", False)
GENERATED_INPUT_DIR = Path(local_path(
    PREPROCESSING_CONFIG.get("output_dir", "data/generated_workflow_inputs")
))
if PREPROCESSING_ENABLED:
    INPUTS.update({
        "split": str(GENERATED_INPUT_DIR / "split.csv"),
        "cv_assignments": str(
            GENERATED_INPUT_DIR / "cv_test_assignments.csv.gz"),
        "train_structures": str(GENERATED_INPUT_DIR / "train.csv.gz"),
        "train_patterns": str(GENERATED_INPUT_DIR / "train.mat.gz"),
        "test_patterns": str(GENERATED_INPUT_DIR / "test.mat.gz"),
    })
OUTPUT_DIR = Path(local_path(config.get("output_dir", "results")))
WORK_DIR = Path(local_path(config.get("work_dir", ".snakemake/isopleth")))
FOLDS = [str(fold) for fold in config.get("folds", range(1, 11))]
TRAINING = config["training"]
METALEARNING = config.get("metalearning", {})
METALEARNING_ENABLED = METALEARNING.get("enabled", False)
RESOURCES = config.get("resources", {})

FINAL_MODEL = str(OUTPUT_DIR / "isopleth_model.pt")
FINAL_LOSS = str(OUTPUT_DIR / "final_training_loss.csv")
FINAL_PREDICTIONS = str(OUTPUT_DIR / "isopleth_predictions.csv.gz")
CV_EVALUATION = str(OUTPUT_DIR / "cross_validation.csv")
FINAL_META_PREDICTIONS = str(OUTPUT_DIR / "metalearning_predictions.csv.gz")
META_CV_EVALUATION = str(
    OUTPUT_DIR / "metalearning_cross_validation.csv")
PREPARED = str(WORK_DIR / "prepared")
META_TRAIN_COSINES = str(
    WORK_DIR / "combined" / "metalearning_train_cosines.csv.gz")

WORKFLOW_TARGETS = [
    FINAL_MODEL,
    FINAL_LOSS,
    FINAL_PREDICTIONS,
    CV_EVALUATION,
]
if METALEARNING_ENABLED:
    WORKFLOW_TARGETS.extend([
        FINAL_META_PREDICTIONS,
        META_CV_EVALUATION,
    ])


def fold_file(wildcards, filename):
    return str(Path(PREPARED) / "cv" / f"fold_{wildcards.fold}" / filename)


def training_arguments():
    names = [
        "fp_algo", "fp_type", "fp_length", "fp_radius", "emb_dim",
        "fp_hidden_dim", "patt_hidden_dim1", "patt_hidden_dim2",
        "tau_clip", "batch_size", "seed", "learning_rate", "patience",
        "ema_alpha", "max_grad_norm", "max_steps", "max_epochs",
        "log_every_steps", "log_every_epochs",
    ]
    return " ".join(f"--{name} {TRAINING[name]}" for name in names)


TRAINING_ARGUMENTS = training_arguments()


rule all:
    input:
        WORKFLOW_TARGETS


if PREPROCESSING_ENABLED:

    rule preprocess_data:
        input:
            peaks=PREPROCESSING["peaks"],
            standards=PREPROCESSING["standards"],
            hmdb=PREPROCESSING["hmdb"],
            labelling_matrix=PREPROCESSING["labelling_matrix"],
            fold_assignments=PREPROCESSING["fold_assignments"],
        output:
            split=INPUTS["split"],
            cv_assignments=INPUTS["cv_assignments"],
            train_structures=INPUTS["train_structures"],
            train_patterns=INPUTS["train_patterns"],
            test_patterns=INPUTS["test_patterns"],
        resources:
            mem_mb=int(RESOURCES.get("mem_mb", 64000))
        conda:
            "environment.yaml"
        script:
            "data_preprocessing.py"


rule prepare_inputs:
    input:
        split=INPUTS["split"],
        cv_assignments=INPUTS["cv_assignments"],
        train_structures=INPUTS["train_structures"],
        train_patterns=INPUTS["train_patterns"],
        test_patterns=INPUTS["test_patterns"],
        candidates=INPUTS["candidates"],
        metalearning_train=INPUTS["metalearning_train"],
    output:
        prepared=temp(directory(PREPARED)),
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    script:
        "prepare_inputs.py"


rule train_cv_model:
    input:
        prepared=rules.prepare_inputs.output.prepared,
    output:
        model=temp(str(WORK_DIR / "cv" / "fold_{fold}" / "model.pt")),
        loss=temp(str(WORK_DIR / "cv" / "fold_{fold}" / "loss.csv")),
        predictions=temp(
            str(WORK_DIR / "cv" / "fold_{fold}" / "predictions.csv.gz")),
        meta_predictions=temp(
            str(WORK_DIR / "cv" / "fold_{fold}" /
                "meta_predictions.csv.gz")),
        timing=temp(str(WORK_DIR / "cv" / "fold_{fold}" / "time.csv")),
    params:
        train_mat=lambda wc: fold_file(wc, "train.mat.csv.gz"),
        train_meta=lambda wc: fold_file(wc, "train.csv.gz"),
        test_mat=lambda wc: fold_file(wc, "test.mat.csv.gz"),
        candidates=lambda wc: fold_file(wc, "test_candidates.csv.gz"),
        meta_candidates=lambda wc: fold_file(wc, "meta_candidates.csv.gz"),
        training=TRAINING_ARGUMENTS,
        script=str(BASE / "train_isopleth.py"),
    threads:
        int(RESOURCES.get("threads", 1))
    resources:
        gpu=int(RESOURCES.get("gpu", 1)),
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    wildcard_constraints:
        fold="|".join(FOLDS)
    shell:
        """
        python {params.script} {params.training} \
          --train_mat_file {params.train_mat} \
          --train_file {params.train_meta} \
          --test_mat_file {params.test_mat} \
          --test_cands_file {params.candidates} \
          --meta_test_cands_file {params.meta_candidates} \
          --model_file {output.model} \
          --loss_file {output.loss} \
          --pred_file {output.predictions} \
          --meta_pred_file {output.meta_predictions} \
          --time_file {output.timing}
        """


rule train_final_model:
    input:
        prepared=rules.prepare_inputs.output.prepared,
    output:
        model=FINAL_MODEL,
        loss=FINAL_LOSS,
        predictions=temp(str(OUTPUT_DIR / ".unseen_predictions.csv.gz")),
        timing=temp(str(OUTPUT_DIR / ".final_time.csv")),
    params:
        train_mat=str(Path(PREPARED) / "final" / "train.mat.csv.gz"),
        train_meta=str(Path(PREPARED) / "final" / "train.csv.gz"),
        test_mat=str(Path(PREPARED) / "final" / "test.mat.csv.gz"),
        candidates=str(Path(PREPARED) / "final" / "test_candidates.csv.gz"),
        training=TRAINING_ARGUMENTS,
        script=str(BASE / "train_isopleth.py"),
    threads:
        int(RESOURCES.get("threads", 1))
    resources:
        gpu=int(RESOURCES.get("gpu", 1)),
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    shell:
        """
        python {params.script} {params.training} \
          --train_mat_file {params.train_mat} \
          --train_file {params.train_meta} \
          --test_mat_file {params.test_mat} \
          --test_cands_file {params.candidates} \
          --model_file {output.model} \
          --loss_file {output.loss} \
          --pred_file {output.predictions} \
          --time_file {output.timing}
        """


rule combine_predictions:
    input:
        folds=expand(
            str(WORK_DIR / "cv" / "fold_{fold}" / "predictions.csv.gz"),
            fold=FOLDS),
        meta_folds=expand(
            str(WORK_DIR / "cv" / "fold_{fold}" /
                "meta_predictions.csv.gz"),
            fold=FOLDS),
        final=rules.train_final_model.output.predictions,
        split=INPUTS["split"],
        candidates=INPUTS["candidates"],
        metalearning_train=INPUTS["metalearning_train"],
    output:
        predictions=FINAL_PREDICTIONS,
        cv_predictions=temp(str(WORK_DIR / "combined" / "cv_predictions.csv.gz")),
        cv_truth=temp(str(WORK_DIR / "combined" / "cv_truth.csv.gz")),
        metalearning_train_cosines=temp(META_TRAIN_COSINES),
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    script:
        "combine_predictions.py"


rule evaluate_cross_validation:
    input:
        truth=rules.combine_predictions.output.cv_truth,
        predictions=rules.combine_predictions.output.cv_predictions,
    output:
        evaluation=CV_EVALUATION,
    params:
        script=str(BASE / "evaluate.py"),
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    shell:
        """
        python {params.script} \
          --test_file {input.truth} \
          --pred_file {input.predictions} \
          --eval_file {output.evaluation}
        """


rule train_cv_metalearning:
    input:
        prepared=rules.prepare_inputs.output.prepared,
        train_cosines=rules.combine_predictions.output.metalearning_train_cosines,
        test_cosines=rules.combine_predictions.output.predictions,
    output:
        predictions=temp(
            str(WORK_DIR / "meta" / "cv" / "fold_{fold}" /
                "predictions.csv.gz")),
        coefficients=temp(
            str(WORK_DIR / "meta" / "cv" / "fold_{fold}" /
                "coefficients.csv")),
    params:
        train_features=lambda wc: fold_file(wc, "train_features.csv.gz"),
        test_features=lambda wc: fold_file(wc, "test_features.csv.gz"),
        seed=int(METALEARNING.get("seed", 0)),
        n_estimators=int(METALEARNING.get("n_estimators", 100)),
        script=str(BASE / "train_metalearning.py"),
    threads:
        int(RESOURCES.get("threads", 1))
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    wildcard_constraints:
        fold="|".join(FOLDS)
    shell:
        """
        python {params.script} \
          --train_feat_file {params.train_features} \
          --test_feat_file {params.test_features} \
          --train_cosine_file {input.train_cosines} \
          --test_cosine_file {input.test_cosines} \
          --pred_file {output.predictions} \
          --coef_file {output.coefficients} \
          --seed {params.seed} \
          --n_estimators {params.n_estimators}
        """


rule train_final_metalearning:
    input:
        prepared=rules.prepare_inputs.output.prepared,
        train_cosines=rules.combine_predictions.output.metalearning_train_cosines,
        test_cosines=rules.combine_predictions.output.predictions,
    output:
        predictions=temp(
            str(WORK_DIR / "meta" / "final" / "predictions.csv.gz")),
        coefficients=temp(
            str(WORK_DIR / "meta" / "final" / "coefficients.csv")),
    params:
        train_features=str(Path(PREPARED) / "final" /
                           "train_features.csv.gz"),
        test_features=str(Path(PREPARED) / "final" /
                          "test_features.csv.gz"),
        seed=int(METALEARNING.get("seed", 0)),
        n_estimators=int(METALEARNING.get("n_estimators", 100)),
        script=str(BASE / "train_metalearning.py"),
    threads:
        int(RESOURCES.get("threads", 1))
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    shell:
        """
        python {params.script} \
          --train_feat_file {params.train_features} \
          --test_feat_file {params.test_features} \
          --train_cosine_file {input.train_cosines} \
          --test_cosine_file {input.test_cosines} \
          --pred_file {output.predictions} \
          --coef_file {output.coefficients} \
          --seed {params.seed} \
          --n_estimators {params.n_estimators}
        """


rule combine_metalearning:
    input:
        folds=expand(
            str(WORK_DIR / "meta" / "cv" / "fold_{fold}" /
                "predictions.csv.gz"),
            fold=FOLDS),
        final=rules.train_final_metalearning.output.predictions,
        split=INPUTS["split"],
        candidates=INPUTS["candidates"],
    output:
        predictions=FINAL_META_PREDICTIONS,
        cv_predictions=temp(
            str(WORK_DIR / "meta" / "combined" /
                "cv_predictions.csv.gz")),
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    script:
        "combine_metalearning.py"


rule evaluate_meta_cross_validation:
    input:
        truth=rules.combine_predictions.output.cv_truth,
        predictions=rules.combine_metalearning.output.cv_predictions,
    output:
        evaluation=META_CV_EVALUATION,
    params:
        script=str(BASE / "evaluate.py"),
    resources:
        mem_mb=int(RESOURCES.get("mem_mb", 64000))
    conda:
        "environment.yaml"
    shell:
        """
        python {params.script} \
          --test_file {input.truth} \
          --pred_file {input.predictions} \
          --eval_file {output.evaluation}
        """
