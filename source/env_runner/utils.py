import re
import os


def sort_checkpoint_files_by_fitness(directory_path, descending=True):
    """fitness 기준으로 checkpoint 파일들을 정렬"""

    # .pkl 파일들 중 checkpoint 파일만 필터링
    pkl_files = [
        f for f in os.listdir(directory_path) if f.endswith(".pkl") and "fitness_" in f
    ]

    def extract_fitness(filename):
        """파일명에서 fitness 값 추출"""
        # fitness_-83.33 또는 fitness_123.45 패턴 찾기
        match = re.search(r"fitness_(-?\d+\.?\d*)", filename)
        if match:
            return float(match.group(1))
        return float("-inf")  # fitness를 찾을 수 없으면 최저값

    # fitness 기준으로 정렬 (기본: 내림차순 = 높은 점수부터)
    sorted_files = sorted(pkl_files, key=extract_fitness, reverse=descending)

    return sorted_files


def sort_checkpoint_files_by_reward(directory_path, descending=True):
    """reward 기준으로 checkpoint 파일들을 정렬"""

    # .pth 파일들 중 checkpoint 파일만 필터링
    pth_files = [
        f for f in os.listdir(directory_path) if f.endswith(".pth") and "reward_" in f
    ]

    def extract_reward(filename):
        """파일명에서 reward 값 추출"""
        # reward_0_490 패턴 (언더스코어로 구분된 소수점) 먼저 찾기
        match = re.search(r"reward_(-?\d+)_(\d+)", filename)
        if match:
            integer_part = match.group(1)
            decimal_part = match.group(2)
            return float(f"{integer_part}.{decimal_part}")

        # reward_-83.33 또는 reward_123.45 패턴 (기존 패턴) 찾기
        match = re.search(r"reward_(-?\d+\.?\d*)", filename)
        if match:
            return float(match.group(1))

        return float("-inf")  # reward를 찾을 수 없으면 최저값

    # reward 기준으로 정렬 (기본: 내림차순 = 높은 점수부터)
    sorted_files = sorted(pth_files, key=extract_reward, reverse=descending)

    return sorted_files
