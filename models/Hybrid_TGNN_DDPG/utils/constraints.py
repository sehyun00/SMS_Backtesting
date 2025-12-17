"""
Portfolio Constraints Utilities
포트폴리오 제약 조건 관리
"""

import torch
import numpy as np


def enforce_weight_constraints(
    weights, min_weight=0.05, max_weight=0.20, max_iter=10, eps=1e-4
):
    """
    반복적 투영으로 포트폴리오 가중치 제약 조건 강제

    보장사항:
    1. min_weight <= w_i <= max_weight (모든 종목)
    2. sum(w) = 1.0
    3. 정규화를 통한 우회 방지

    Args:
        weights: (Batch, N) 또는 (N,) 가중치 텐서
        min_weight: 최소 비중
        max_weight: 최대 비중
        max_iter: 최대 반복 횟수
        eps: 수렴 허용 오차

    Returns:
        constrained_weights: 제약 조건을 만족하는 가중치
    """
    if isinstance(weights, np.ndarray):
        weights = torch.FloatTensor(weights)

    original_shape = weights.shape
    if len(original_shape) == 1:
        weights = weights.unsqueeze(0)

    batch_size, n_stocks = weights.shape

    for iteration in range(max_iter):
        # Step 1: [min_weight, max_weight]로 클램핑
        weights_clamped = torch.clamp(weights, min_weight, max_weight)

        # Step 2: 현재 합계 확인
        current_sum = weights_clamped.sum(dim=-1, keepdim=True)

        # Step 3: 합계가 1.0에 가까우면 완료
        if torch.allclose(current_sum, torch.ones_like(current_sum), atol=eps):
            weights = weights_clamped
            break

        # Step 4: 초과/부족분 재분배
        deficit = 1.0 - current_sum  # (Batch, 1)

        # 증가/감소 필요 여부 확인
        need_increase = deficit > 0  # (Batch, 1)

        # 조정 가능한 종목 찾기
        room_to_grow = max_weight - weights_clamped  # (Batch, N)
        room_to_shrink = weights_clamped - min_weight  # (Batch, N)

        # 증가용: 성장 여지가 있는 종목에 분배
        total_room_grow = room_to_grow.sum(dim=-1, keepdim=True)
        adjustment_grow = torch.where(
            total_room_grow > eps,
            deficit * (room_to_grow / (total_room_grow + 1e-8)),
            deficit / n_stocks,
        )

        # 감소용: 축소 여지가 있는 종목에서 차감
        total_room_shrink = room_to_shrink.sum(dim=-1, keepdim=True)
        adjustment_shrink = torch.where(
            total_room_shrink > eps,
            deficit * (room_to_shrink / (total_room_shrink + 1e-8)),
            deficit / n_stocks,
        )

        # 적절한 조정 적용
        adjustment = torch.where(need_increase, adjustment_grow, adjustment_shrink)
        weights = weights_clamped + adjustment

    # 최종 안전 장치: 클램핑 및 정규화
    weights = torch.clamp(weights, min_weight, max_weight)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # 원래 shape으로 복원
    if len(original_shape) == 1:
        weights = weights.squeeze(0)

    return weights


def check_constraint_violation(
    weights, min_weight=0.05, max_weight=0.20, sum_tolerance=1e-3
):
    """
    제약 조건 위반 여부 확인

    Args:
        weights: (Batch, N) 또는 (N,) 가중치
        min_weight: 최소 비중
        max_weight: 최대 비중
        sum_tolerance: 합계 허용 오차

    Returns:
        violations: dict with violation details
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    if len(weights.shape) == 1:
        weights = weights.reshape(1, -1)

    violations = {
        "below_min": [],
        "above_max": [],
        "sum_violations": [],
    }

    for i, w in enumerate(weights):
        # 최소/최대 비중 위반
        below_min_idx = np.where(w < min_weight)[0]
        above_max_idx = np.where(w > max_weight)[0]

        if len(below_min_idx) > 0:
            violations["below_min"].append((i, below_min_idx.tolist()))

        if len(above_max_idx) > 0:
            violations["above_max"].append((i, above_max_idx.tolist()))

        # 합계 위반
        weight_sum = np.sum(w)
        if abs(weight_sum - 1.0) > sum_tolerance:
            violations["sum_violations"].append((i, weight_sum))

    return violations


def calculate_concentration_metrics(weights):
    """
    집중도 메트릭 계산

    Args:
        weights: (Batch, N) 또는 (N,) 가중치

    Returns:
        metrics: dict with herfindahl_index, effective_stocks, max_weight
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    if len(weights.shape) == 1:
        weights = weights.reshape(1, -1)

    metrics_list = []

    for w in weights:
        # Herfindahl Index (HHI)
        hhi = np.sum(w**2)

        # Effective number of stocks
        effective_stocks = 1.0 / hhi if hhi > 0 else len(w)

        # Maximum weight
        max_weight = np.max(w)

        # Entropy (다양성)
        entropy = -np.sum(w * np.log(w + 1e-10))
        max_entropy = np.log(len(w))
        normalized_entropy = entropy / max_entropy

        metrics_list.append(
            {
                "herfindahl_index": hhi,
                "effective_stocks": effective_stocks,
                "max_weight": max_weight,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
            }
        )

    return metrics_list[0] if len(weights) == 1 else metrics_list
