"""
Portfolio Constraints
포트폴리오 제약 조건
"""

import numpy as np


def apply_concentration_limit(weights, max_weight=0.15):
    """
    단일 종목 최대 비중 제한 (Concentration Limit)

    과도한 집중을 방지하여 리스크를 분산시킵니다.
    README 기준: 단일 종목 최대 15% 비중

    Args:
        weights: (N,) 포트폴리오 가중치 numpy array
        max_weight: 단일 종목 최대 허용 비중 (0.15 = 15%)

    Returns:
        constrained_weights: (N,) 제약이 적용된 가중치 (sum=1 보장)

    Example:
        >>> weights = np.array([0.5, 0.3, 0.2])  # 초기 가중치
        >>> constrained = apply_concentration_limit(weights, max_weight=0.15)
        >>> # 결과: [0.15, 0.15, 0.2] -> 정규화 -> [0.3, 0.3, 0.4]
    """
    # 가중치를 numpy array로 변환
    weights = np.array(weights, dtype=np.float32)

    # 최대 비중으로 클립
    clipped_weights = np.clip(weights, 0, max_weight)

    # 합이 1이 되도록 정규화
    weight_sum = clipped_weights.sum()

    if weight_sum > 1e-8:
        normalized_weights = clipped_weights / weight_sum
    else:
        # 모든 가중치가 0이면 균등 분배
        normalized_weights = np.ones_like(weights) / len(weights)

    return normalized_weights


def apply_min_weight_threshold(weights, min_weight=0.01):
    """
    최소 비중 임계값 적용

    너무 작은 비중을 0으로 설정하여 거래 비용을 절감합니다.

    Args:
        weights: (N,) 포트폴리오 가중치 numpy array
        min_weight: 최소 허용 비중 (1% 미만은 0으로 처리)

    Returns:
        thresholded_weights: (N,) 임계값이 적용된 가중치 (sum=1 보장)
    """
    weights = np.array(weights, dtype=np.float32)

    # 임계값 미만은 0으로 설정
    thresholded = np.where(weights < min_weight, 0, weights)

    # 정규화
    weight_sum = thresholded.sum()

    if weight_sum > 1e-8:
        normalized = thresholded / weight_sum
    else:
        # 모두 0이면 균등 분배
        normalized = np.ones_like(weights) / len(weights)

    return normalized


def apply_long_only_constraint(weights):
    """
    Long-only 제약 (매도 금지, 가중치 >= 0)

    Args:
        weights: (N,) 포트폴리오 가중치 numpy array

    Returns:
        long_only_weights: (N,) 양수 제약이 적용된 가중치 (sum=1 보장)
    """
    weights = np.array(weights, dtype=np.float32)

    # 음수 가중치를 0으로 클립
    long_only = np.clip(weights, 0, None)

    # 정규화
    weight_sum = long_only.sum()

    if weight_sum > 1e-8:
        normalized = long_only / weight_sum
    else:
        normalized = np.ones_like(weights) / len(weights)

    return normalized
