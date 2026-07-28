"""Full optimization objective assembly (Phase G)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.data.validation import DataIntegrityError
from epistemic_sycophancy.objective.aggregation import question_macro_mean
from epistemic_sycophancy.objective.losses import (
    baseline_relative_hinge,
    logistic_margin_loss,
)


@dataclass(frozen=True)
class ObjectiveResult:
    """Scalar objective components and weighted total."""

    l_resist: float
    l_recover: float
    l_behavior: float
    l_neutral: float
    l_correct: float
    l_beta: float
    l_total: float


def resistance_prompt_losses(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> dict[object, list[float]]:
    """Return φ(M) for each IB prompt of each question in Q+.

    Only q∈Q+ and IB variants are eligible (OBJ-001).
    """
    q_plus_set = frozenset(q_plus)
    return {
        question_id: [
            logistic_margin_loss(float(margin), tau=tau) for margin in margins
        ]
        for question_id, margins in ib_margins_by_question.items()
        if question_id in q_plus_set
    }


def resistance_loss(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> float:
    """Question-macro resistance: mean within question, then across Q+."""
    return question_macro_mean(
        resistance_prompt_losses(
            ib_margins_by_question=ib_margins_by_question,
            q_plus=q_plus,
            tau=tau,
        )
    )


def recovery_prompt_losses(
    *,
    cb_margins_by_question: Mapping[object, Sequence[float]],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> dict[object, list[float]]:
    """Return φ(M) for each CB prompt of each question in Q-."""
    q_minus_set = frozenset(q_minus)
    return {
        question_id: [
            logistic_margin_loss(float(margin), tau=tau) for margin in margins
        ]
        for question_id, margins in cb_margins_by_question.items()
        if question_id in q_minus_set
    }


def recovery_loss(
    *,
    cb_margins_by_question: Mapping[object, Sequence[float]],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> float:
    """Question-macro recovery: mean within question, then across Q-."""
    return question_macro_mean(
        recovery_prompt_losses(
            cb_margins_by_question=cb_margins_by_question,
            q_minus=q_minus,
            tau=tau,
        )
    )


def behavioral_loss(
    *,
    l_resist: float,
    l_recover: float,
    w_r: float,
    w_u: float,
) -> float:
    """L_behavior = w_R L_resist + w_U L_recover (explicit weights, not subset sizes)."""
    return float(w_r) * float(l_resist) + float(w_u) * float(l_recover)


def neutral_question_penalties(
    *,
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    delta_n: float,
) -> dict[object, float]:
    """Per-question neutral hinge d_q,N = [M0 - M(β) - δ_N]_+."""
    return {
        question_id: float(
            baseline_relative_hinge(
                baseline_margin=float(baseline_neutral_margins[question_id]),
                current_margin=float(current_neutral_margins[question_id]),
                delta=delta_n,
            )
        )
        for question_id in baseline_neutral_margins
    }


def neutral_preservation_loss(
    *,
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    delta_n: float,
) -> float:
    """L_neutral = (1/|Q|) Σ_q d_q,N over the full optimization question set."""
    penalties = neutral_question_penalties(
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        delta_n=delta_n,
    )
    if not penalties:
        raise ValueError("neutral_preservation_loss requires at least one question")
    return sum(penalties.values()) / float(len(penalties))


def correct_belief_question_penalties(
    *,
    baseline_cb_margins: Mapping[object, Sequence[float]],
    current_cb_margins: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    delta_c: float,
) -> dict[object, list[float]]:
    """Per-variant CB hinges for each q∈Q+: [M0 - M(β) - δ_C]_+."""
    q_plus_set = frozenset(q_plus)
    result: dict[object, list[float]] = {}
    for question_id in q_plus_set:
        baselines = baseline_cb_margins[question_id]
        currents = current_cb_margins[question_id]
        if len(baselines) != len(currents):
            raise ValueError(
                f"baseline/current CB margin length mismatch for {question_id!r}"
            )
        result[question_id] = [
            float(
                baseline_relative_hinge(
                    baseline_margin=float(b0),
                    current_margin=float(m),
                    delta=delta_c,
                )
            )
            for b0, m in zip(baselines, currents, strict=True)
        ]
    return result


def correct_belief_preservation_loss(
    *,
    baseline_cb_margins: Mapping[object, Sequence[float]],
    current_cb_margins: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    delta_c: float,
) -> float:
    """Question-macro correct-belief hinge over Q+."""
    return question_macro_mean(
        correct_belief_question_penalties(
            baseline_cb_margins=baseline_cb_margins,
            current_cb_margins=current_cb_margins,
            q_plus=q_plus,
            delta_c=delta_c,
        )
    )


def coefficient_regularizer(*, beta: Sequence[float]) -> float:
    """L_beta = mean_j |β_j| over normalized coefficients."""
    if not beta:
        raise ValueError("beta must be non-empty")
    return sum(abs(float(b)) for b in beta) / float(len(beta))


def validate_objective_rows(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    cb_margins_by_question: Mapping[object, Sequence[float]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
) -> None:
    """Raise DataIntegrityError when required IB/CB/N rows are missing (DEC-028)."""
    q_plus_set = frozenset(q_plus)
    q_minus_set = frozenset(q_minus)

    for question_id in baseline_neutral_margins:
        if question_id not in current_neutral_margins:
            raise DataIntegrityError(
                f"missing current neutral margin for question {question_id!r}"
            )

    for question_id in q_plus_set:
        ib = ib_margins_by_question.get(question_id)
        if not ib:
            raise DataIntegrityError(
                f"missing IB variants for Q+ question {question_id!r}"
            )
        cb = cb_margins_by_question.get(question_id)
        if not cb:
            raise DataIntegrityError(
                f"missing CB variants for Q+ question {question_id!r}"
            )
        baseline_cb = baseline_cb_margins.get(question_id)
        if not baseline_cb:
            raise DataIntegrityError(
                f"missing baseline CB margins for Q+ question {question_id!r}"
            )

    for question_id in q_minus_set:
        cb = cb_margins_by_question.get(question_id)
        if not cb:
            raise DataIntegrityError(
                f"missing CB variants for Q- question {question_id!r}"
            )


def evaluate_objective(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    cb_margins_by_question: Mapping[object, Sequence[float]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    beta: Sequence[float],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> ObjectiveResult:
    """Assemble all objective components and the weighted total."""
    validate_objective_rows(
        ib_margins_by_question=ib_margins_by_question,
        cb_margins_by_question=cb_margins_by_question,
        baseline_cb_margins=baseline_cb_margins,
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        q_plus=q_plus,
        q_minus=q_minus,
    )
    l_resist = resistance_loss(
        ib_margins_by_question=ib_margins_by_question,
        q_plus=q_plus,
        tau=tau,
    )
    l_recover = recovery_loss(
        cb_margins_by_question=cb_margins_by_question,
        q_minus=q_minus,
        tau=tau,
    )
    l_behavior = behavioral_loss(
        l_resist=l_resist,
        l_recover=l_recover,
        w_r=w_r,
        w_u=w_u,
    )
    l_neutral = neutral_preservation_loss(
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        delta_n=delta_n,
    )
    l_correct = correct_belief_preservation_loss(
        baseline_cb_margins=baseline_cb_margins,
        current_cb_margins=cb_margins_by_question,
        q_plus=q_plus,
        delta_c=delta_c,
    )
    l_beta = coefficient_regularizer(beta=beta)
    l_total = (
        l_behavior
        + float(lambda_n) * l_neutral
        + float(lambda_c) * l_correct
        + float(lambda_beta) * l_beta
    )
    return ObjectiveResult(
        l_resist=l_resist,
        l_recover=l_recover,
        l_behavior=l_behavior,
        l_neutral=l_neutral,
        l_correct=l_correct,
        l_beta=l_beta,
        l_total=l_total,
    )


def _affine_margin_tensors(
    *,
    const: Sequence[float],
    jac_rows: Sequence[object],
    beta: object,
) -> list[object]:
    import torch

    return [
        float(c) + torch.dot(g.to(dtype=beta.dtype), beta)
        for c, g in zip(const, jac_rows, strict=True)
    ]


def _softplus_neg_margin(margin: object, *, tau: float) -> object:
    import torch

    return torch.nn.functional.softplus(-margin / float(tau))


def _hinge_excess(baseline: float, current: object, *, delta: float) -> object:
    import torch

    return torch.relu(float(baseline) - current - float(delta))


def _sum_question_means_phi(
    *,
    question_ids: Sequence[object],
    eligible: frozenset[object],
    margin_const: Mapping[object, Sequence[float]],
    margin_jac: Mapping[object, Sequence[object]],
    beta: object,
    tau: float,
) -> object:
    import torch

    means: list[object] = []
    for question_id in question_ids:
        if question_id not in eligible:
            continue
        margins = _affine_margin_tensors(
            const=margin_const[question_id],
            jac_rows=margin_jac[question_id],
            beta=beta,
        )
        losses = [_softplus_neg_margin(m, tau=tau) for m in margins]
        means.append(torch.stack(losses).mean())
    if not means:
        return torch.zeros((), dtype=beta.dtype, device=beta.device)
    return torch.stack(means).sum()


def _sum_question_means_hinge(
    *,
    question_ids: Sequence[object],
    eligible: frozenset[object],
    baseline_margins: Mapping[object, Sequence[float]],
    margin_const: Mapping[object, Sequence[float]],
    margin_jac: Mapping[object, Sequence[object]],
    beta: object,
    delta: float,
) -> object:
    import torch

    means: list[object] = []
    for question_id in question_ids:
        if question_id not in eligible:
            continue
        currents = _affine_margin_tensors(
            const=margin_const[question_id],
            jac_rows=margin_jac[question_id],
            beta=beta,
        )
        baselines = baseline_margins[question_id]
        hinges = [
            _hinge_excess(float(b0), m, delta=delta)
            for b0, m in zip(baselines, currents, strict=True)
        ]
        means.append(torch.stack(hinges).mean())
    if not means:
        return torch.zeros((), dtype=beta.dtype, device=beta.device)
    return torch.stack(means).sum()


def _sum_neutral_hinges(
    *,
    question_ids: Sequence[object],
    baseline_neutral_margins: Mapping[object, float],
    neutral_margin_const: Mapping[object, float],
    neutral_margin_jac: Mapping[object, object],
    beta: object,
    delta_n: float,
) -> object:
    import torch

    terms: list[object] = []
    for question_id in question_ids:
        current = float(neutral_margin_const[question_id]) + torch.dot(
            neutral_margin_jac[question_id].to(dtype=beta.dtype), beta
        )
        terms.append(
            _hinge_excess(
                float(baseline_neutral_margins[question_id]),
                current,
                delta=delta_n,
            )
        )
    if not terms:
        return torch.zeros((), dtype=beta.dtype, device=beta.device)
    return torch.stack(terms).sum()


def _objective_tensor_from_question_ids(
    *,
    beta: object,
    question_ids: Sequence[object],
    ib_margin_const: Mapping[object, Sequence[float]],
    ib_margin_jac: Mapping[object, Sequence[object]],
    cb_margin_const: Mapping[object, Sequence[float]],
    cb_margin_jac: Mapping[object, Sequence[object]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    neutral_margin_const: Mapping[object, float],
    neutral_margin_jac: Mapping[object, object],
    q_plus: frozenset[object],
    q_minus: frozenset[object],
    n_q_plus: int,
    n_q_minus: int,
    n_q: int,
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
    include_beta_penalty: bool,
) -> object:
    import torch

    resist_sum = _sum_question_means_phi(
        question_ids=question_ids,
        eligible=q_plus,
        margin_const=ib_margin_const,
        margin_jac=ib_margin_jac,
        beta=beta,
        tau=tau,
    )
    recover_sum = _sum_question_means_phi(
        question_ids=question_ids,
        eligible=q_minus,
        margin_const=cb_margin_const,
        margin_jac=cb_margin_jac,
        beta=beta,
        tau=tau,
    )
    neutral_sum = _sum_neutral_hinges(
        question_ids=question_ids,
        baseline_neutral_margins=baseline_neutral_margins,
        neutral_margin_const=neutral_margin_const,
        neutral_margin_jac=neutral_margin_jac,
        beta=beta,
        delta_n=delta_n,
    )
    correct_sum = _sum_question_means_hinge(
        question_ids=question_ids,
        eligible=q_plus,
        baseline_margins=baseline_cb_margins,
        margin_const=cb_margin_const,
        margin_jac=cb_margin_jac,
        beta=beta,
        delta=delta_c,
    )
    l_resist = resist_sum / float(n_q_plus)
    l_recover = recover_sum / float(n_q_minus)
    l_behavior = float(w_r) * l_resist + float(w_u) * l_recover
    l_neutral = neutral_sum / float(n_q)
    l_correct = correct_sum / float(n_q_plus)
    l_total = (
        l_behavior
        + float(lambda_n) * l_neutral
        + float(lambda_c) * l_correct
    )
    if include_beta_penalty:
        l_beta = beta.abs().mean()
        l_total = l_total + float(lambda_beta) * l_beta
    return l_total


def evaluate_objective_with_grad(
    *,
    beta: object,
    ib_margin_const: Mapping[object, Sequence[float]],
    ib_margin_jac: Mapping[object, Sequence[object]],
    cb_margin_const: Mapping[object, Sequence[float]],
    cb_margin_jac: Mapping[object, Sequence[object]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    neutral_margin_const: Mapping[object, float],
    neutral_margin_jac: Mapping[object, object],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> tuple[float, list[float]]:
    """Full-split objective and ∂L/∂β under affine margins (DEC-027)."""
    import torch

    q_plus_set = frozenset(q_plus)
    q_minus_set = frozenset(q_minus)
    all_questions = tuple(baseline_neutral_margins.keys())
    beta_leaf = beta.detach().to(dtype=torch.float64).clone().requires_grad_(True)
    loss = _objective_tensor_from_question_ids(
        beta=beta_leaf,
        question_ids=all_questions,
        ib_margin_const=ib_margin_const,
        ib_margin_jac=ib_margin_jac,
        cb_margin_const=cb_margin_const,
        cb_margin_jac=cb_margin_jac,
        baseline_cb_margins=baseline_cb_margins,
        baseline_neutral_margins=baseline_neutral_margins,
        neutral_margin_const=neutral_margin_const,
        neutral_margin_jac=neutral_margin_jac,
        q_plus=q_plus_set,
        q_minus=q_minus_set,
        n_q_plus=len(q_plus_set),
        n_q_minus=len(q_minus_set),
        n_q=len(all_questions),
        tau=tau,
        w_r=w_r,
        w_u=w_u,
        delta_n=delta_n,
        delta_c=delta_c,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
        include_beta_penalty=True,
    )
    loss.backward()
    assert beta_leaf.grad is not None
    return float(loss.detach()), [float(x) for x in beta_leaf.grad.detach()]


def accumulate_objective_batches(
    *,
    beta: object,
    question_batches: Sequence[Sequence[object]],
    ib_margin_const: Mapping[object, Sequence[float]],
    ib_margin_jac: Mapping[object, Sequence[object]],
    cb_margin_const: Mapping[object, Sequence[float]],
    cb_margin_jac: Mapping[object, Sequence[object]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    neutral_margin_const: Mapping[object, float],
    neutral_margin_jac: Mapping[object, object],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> tuple[float, list[float]]:
    """Accumulate question-batched objective with full-split denominators (DEC-027)."""
    import torch

    q_plus_set = frozenset(q_plus)
    q_minus_set = frozenset(q_minus)
    all_questions = tuple(baseline_neutral_margins.keys())
    n_q_plus = len(q_plus_set)
    n_q_minus = len(q_minus_set)
    n_q = len(all_questions)

    beta_leaf = beta.detach().to(dtype=torch.float64).clone().requires_grad_(True)
    # Sum numerator contributions across batches, then scale by full denominators.
    # Equivalent to one full-split forward when batches partition all questions.
    resist_sum = torch.zeros((), dtype=torch.float64)
    recover_sum = torch.zeros((), dtype=torch.float64)
    neutral_sum = torch.zeros((), dtype=torch.float64)
    correct_sum = torch.zeros((), dtype=torch.float64)
    for batch in question_batches:
        resist_sum = resist_sum + _sum_question_means_phi(
            question_ids=batch,
            eligible=q_plus_set,
            margin_const=ib_margin_const,
            margin_jac=ib_margin_jac,
            beta=beta_leaf,
            tau=tau,
        )
        recover_sum = recover_sum + _sum_question_means_phi(
            question_ids=batch,
            eligible=q_minus_set,
            margin_const=cb_margin_const,
            margin_jac=cb_margin_jac,
            beta=beta_leaf,
            tau=tau,
        )
        neutral_sum = neutral_sum + _sum_neutral_hinges(
            question_ids=batch,
            baseline_neutral_margins=baseline_neutral_margins,
            neutral_margin_const=neutral_margin_const,
            neutral_margin_jac=neutral_margin_jac,
            beta=beta_leaf,
            delta_n=delta_n,
        )
        correct_sum = correct_sum + _sum_question_means_hinge(
            question_ids=batch,
            eligible=q_plus_set,
            baseline_margins=baseline_cb_margins,
            margin_const=cb_margin_const,
            margin_jac=cb_margin_jac,
            beta=beta_leaf,
            delta=delta_c,
        )

    l_resist = resist_sum / float(n_q_plus)
    l_recover = recover_sum / float(n_q_minus)
    l_behavior = float(w_r) * l_resist + float(w_u) * l_recover
    l_neutral = neutral_sum / float(n_q)
    l_correct = correct_sum / float(n_q_plus)
    l_beta = beta_leaf.abs().mean()
    loss = (
        l_behavior
        + float(lambda_n) * l_neutral
        + float(lambda_c) * l_correct
        + float(lambda_beta) * l_beta
    )
    loss.backward()
    assert beta_leaf.grad is not None
    return float(loss.detach()), [float(x) for x in beta_leaf.grad.detach()]
