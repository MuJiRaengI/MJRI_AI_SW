import os
import pickle
import neat
from typing import List, Dict, Any
import copy


class SaveBestReporter(neat.reporting.BaseReporter):
    """성능이 좋은 상위 N개의 genome을 저장하는 리포터"""

    def __init__(
        self,
        save_dir: str,
        top_n: int = 5,
        config=None,
        debug: bool = True,
    ):
        """
        SaveBestReporter 초기화

        Args:
            save_dir: genome들을 저장할 디렉토리
            top_n: 저장할 상위 genome 개수 (기본: 5)
            debug: 디버그 정보 출력 여부
        """
        self.save_dir = save_dir
        self.top_n = top_n
        self.config = config
        self.generation = 0
        self.debug = debug

        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 최고 성능 genome들을 저장할 리스트
        # 각 항목: {'genome': genome, 'fitness': float, 'generation': int, 'genome_id': int}
        self.top_genomes = []

        print(
            f"🏆 SaveBestReporter initialized: Top {top_n} genomes will be saved when rankings change"
        )

    def start_generation(self, generation):
        """세대 시작 시 호출"""
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """각 세대 평가 후 최고 성능 genome들 업데이트"""

        # 현재 세대의 모든 genome 중에서 fitness가 있는 것들만 수집
        current_genomes = []
        for genome_id, genome in population.items():
            if genome.fitness is not None:
                current_genomes.append(
                    {
                        "genome": copy.deepcopy(genome),  # 깊은 복사로 안전하게 저장
                        "fitness": genome.fitness,
                        "generation": self.generation,
                        "genome_id": genome_id,
                    }
                )

        # 디버깅: 현재 세대의 최고/최저 성능 출력
        if current_genomes:
            current_best = max(current_genomes, key=lambda x: x["fitness"])
            current_worst = min(current_genomes, key=lambda x: x["fitness"])
            print(
                f"Gen {self.generation}: Best={current_best['fitness']:.2f}, Worst={current_worst['fitness']:.2f}, Count={len(current_genomes)}"
            )

            # 새로운 최고 성능 체크
            if (
                not hasattr(self, "best_ever_fitness")
                or current_best["fitness"] > self.best_ever_fitness
            ):
                self.best_ever_fitness = current_best["fitness"]
                print(
                    f"🎉 NEW BEST FITNESS: {current_best['fitness']:.2f} (Generation {self.generation}, ID {current_best['genome_id']})"
                )

        # 현재 최고 성능들과 합치기
        all_genomes = self.top_genomes + current_genomes

        # fitness 기준으로 내림차순 정렬
        all_genomes.sort(key=lambda x: x["fitness"], reverse=True)

        # 상위 N개만 유지
        old_top = self.top_genomes.copy() if self.top_genomes else []
        self.top_genomes = all_genomes[: self.top_n]

        # 순위 변동 확인 및 즉시 저장
        top_changed = False
        if old_top and self.top_genomes:
            if old_top[0]["fitness"] != self.top_genomes[0]["fitness"]:
                print(
                    f"🔄 Top genome changed: {old_top[0]['fitness']:.2f} → {self.top_genomes[0]['fitness']:.2f}"
                )
                top_changed = True
        elif not old_top and self.top_genomes:
            # 처음으로 top genome이 생성된 경우
            top_changed = True
            print(f"🆕 First top genome: {self.top_genomes[0]['fitness']:.2f}")

        # top N 목록이 변경되었거나 처음 5세대인 경우 즉시 저장
        if top_changed or self.generation <= 5:
            if self.debug:
                print(f"💾 Saving updated top genomes at generation {self.generation}")
            self._cleanup_old_files()  # 기존 파일 정리
            self._save_top_genomes()
            if self.debug:
                self._print_status()

        # 디버그 정보
        if self.debug and self.top_genomes and not top_changed:
            print(
                f"🏆 Current top unchanged: {self.top_genomes[0]['fitness']:.2f} (Gen {self.top_genomes[0]['generation']}, ID {self.top_genomes[0]['genome_id']})"
            )

    def _save_top_genomes(self):
        """현재 상위 N개 genome들을 파일로 저장"""

        print(
            f"\n💾 Saving top {len(self.top_genomes)} genomes (Generation {self.generation})..."
        )

        for i, record in enumerate(self.top_genomes, 1):
            filename = f"rank_{i:02d}_fitness_{record['fitness']:.2f}_gen_{record['generation']}_id_{record['genome_id']}.pkl"
            filepath = os.path.join(self.save_dir, filename)

            try:
                # genome과 config, 메타데이터를 함께 저장 (best_genome.pkl과 동일한 형식)
                save_data = {
                    "winner_genome": record["genome"],  # best model과 동일한 키명 사용
                    "config": self.config,  # NEAT 설정 파일
                    "fitness": record["fitness"],
                    "generation": record["generation"],
                    "genome_id": record["genome_id"],
                    "rank": i,
                    "saved_at_generation": self.generation,
                    "is_checkpoint": True,  # checkpoint 표시
                }

                with open(filepath, "wb") as f:
                    pickle.dump(save_data, f)

                if self.debug:
                    print(
                        f"  ✅ Saved rank {i:2d}: Fitness {record['fitness']:8.2f} | Gen {record['generation']:3d} | ID {record['genome_id']:5d}"
                    )
                    print(f"      → {filename}")

            except Exception as e:
                print(f"  ❌ Error saving rank {i}: {e}")

        if self.debug:
            # 저장 후 디렉토리 확인
            existing_files = [
                f for f in os.listdir(self.save_dir) if f.endswith(".pkl")
            ]
            print(f"💽 Total .pkl files in directory after save: {len(existing_files)}")
            for f in sorted(existing_files)[:3]:  # 처음 3개만 표시
                print(f"     {f}")
            if len(existing_files) > 3:
                print(f"     ... and {len(existing_files) - 3} more files")

    def _print_status(self):
        """현재 상태 요약 출력"""
        if not self.top_genomes:
            return

        print(
            f"\n📊 TOP {len(self.top_genomes)} GENOMES SUMMARY (Generation {self.generation}):"
        )
        print("     Rank | Fitness  | Generation | Genome ID")
        print("     -----|----------|------------|----------")

        for i, record in enumerate(self.top_genomes, 1):
            print(
                f"     {i:4d} | {record['fitness']:8.2f} | {record['generation']:10d} | {record['genome_id']:9d}"
            )

    def force_save(self):
        """즉시 저장 (학습 종료시 호출)"""
        print(f"\n🔒 Force saving top genomes at generation {self.generation}...")
        self._save_top_genomes()
        self._print_status()

    def get_top_genomes(self) -> List[Dict[str, Any]]:
        """현재 상위 genome들 정보 반환"""
        return copy.deepcopy(self.top_genomes)

    def load_genome(self, filepath: str) -> Dict[str, Any]:
        """저장된 genome 파일 로드"""
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None

    def get_best_genome(self):
        """현재 최고 성능 genome 반환"""
        if self.top_genomes:
            return self.top_genomes[0]
        return None

    def _cleanup_old_files(self):
        """이전에 저장된 파일들 정리 (현재 top N 제외)"""
        if not os.path.exists(self.save_dir):
            return

        # 기존 모든 .pkl 파일 삭제 (완전히 새로 저장)
        removed_count = 0
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Error removing {filename}: {e}")

        if removed_count > 0 and self.debug:
            print(f"🗑️ Cleaned up {removed_count} old files")

    def cleanup_old_files(self):
        """외부에서 호출 가능한 정리 함수"""
        self._cleanup_old_files()
