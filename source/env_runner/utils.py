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
