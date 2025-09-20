import os
import neat
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from matplotlib.patches import FancyBboxPatch
from .utils import create_network_plot, save_network_plot


class NetworkReporter(neat.reporting.BaseReporter):
    """각 세대 평가 후 모든 종의 최고 성능 네트워크 구조를 시각화하는 리포터"""

    def __init__(self, config, save_dir: str, node_names):
        """
        NetworkReporter 초기화

        Args:
            config: NEAT config 객체
            save_dir: 네트워크 구조 이미지를 저장할 디렉토리
            node_names: 노드 이름 딕셔너리
        """
        self.config = config
        self.save_dir = save_dir
        self.generation = 0

        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 각 종의 최고 성능 기록 {species_id: {'fitness': float, 'genome': genome, 'generation': int}}
        self.species_best = {}

        # 노드 이름 설정
        self.node_names = node_names

    def start_generation(self, generation):
        """세대 시작 시 호출"""
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """각 세대 평가 후 모든 종의 최고 성능 확인 및 네트워크 구조 시각화"""

        print(f"\n🎨 Generation {self.generation} - Drawing Network Structures...")

        for species_id, species_obj in species.species.items():
            # 현재 종의 최고 성능 genome 찾기
            species_genomes = species_obj.members
            current_best = max(
                species_genomes.values(),
                key=lambda g: g.fitness if g.fitness is not None else -999999,
            )

            current_fitness = (
                current_best.fitness if current_best.fitness is not None else -999999
            )

            # 새로운 종이거나 성능이 향상된 경우
            update_needed = False
            is_new_species = species_id not in self.species_best

            if is_new_species:
                print(f"  🎉 NEW SPECIES {species_id} | Fitness: {current_fitness:.2f}")
                update_needed = True
            elif current_fitness > self.species_best[species_id]["fitness"]:
                prev_fitness = self.species_best[species_id]["fitness"]
                print(
                    f"  📈 SPECIES {species_id} IMPROVED | {prev_fitness:.2f} → {current_fitness:.2f}"
                )
                update_needed = True

            if update_needed:
                # 기록 업데이트
                self.species_best[species_id] = {
                    "fitness": current_fitness,
                    "genome": current_best,
                    "generation": self.generation,
                }

                # 네트워크 구조 시각화 및 저장
                self._draw_and_save_network(species_id, current_best, current_fitness)

        # 전체 네트워크 상태 요약
        if self.generation % 5 == 0:  # 5세대마다 요약 출력
            self._print_summary()

    def _draw_and_save_network(self, species_id: int, genome, fitness: float):
        """개별 종의 네트워크 구조를 그리고 저장"""
        try:
            # 파일명 생성
            filename = (
                f"species_{species_id}_gen_{self.generation}_fitness_{fitness:.1f}.png"
            )
            filepath = os.path.join(self.save_dir, filename)

            # 네트워크 구조 그리기
            fig, ax = create_network_plot(genome, self.config, self.node_names)

            # 간단한 제목 추가 (흰색 텍스트에 검은 테두리)
            title = f"S{species_id} G{self.generation} F{fitness:.1f}"
            ax.text(
                5,
                9.7,
                title,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(
                        linewidth=2, foreground="black"
                    )
                ],
            )

            # 파일 저장
            save_network_plot(fig, filepath)

            print(f"    ✅ Network saved: {filename}")

        except Exception as e:
            print(f"    ❌ Error saving species {species_id}: {e}")

    def _print_summary(self):
        """현재 상태 요약 출력"""
        print(f"\n📊 NETWORK SUMMARY (Generation {self.generation}):")
        print(f"   Save Directory: {self.save_dir}")
        print(f"   Active Species: {len(self.species_best)}")

        for species_id, record in self.species_best.items():
            genome = record["genome"]
            nodes = len(genome.nodes)
            connections = len([c for c in genome.connections.values() if c.enabled])
            print(
                f"   Species {species_id}: Fitness={record['fitness']:.2f} | Nodes={nodes} | Connections={connections} (Gen {record['generation']})"
            )

    def get_best_networks_info(self):
        """현재까지의 최고 성능 네트워크들 정보 반환"""
        return self.species_best.copy()

    def save_all_current_networks(self, prefix: str = "final"):
        """현재 모든 종의 최고 성능 네트워크를 특별한 이름으로 저장"""
        print(f"\n💾 Saving all current best networks with prefix '{prefix}'...")

        for species_id, record in self.species_best.items():
            genome = record["genome"]
            fitness = record["fitness"]
            generation = record["generation"]

            filename = f"{prefix}_species_{species_id}_gen_{generation}_fitness_{fitness:.1f}.png"
            filepath = os.path.join(self.save_dir, filename)

            try:
                fig, ax = create_network_plot(genome, self.config, self.node_names)
                title = f"{prefix.upper()} S{species_id} F{fitness:.1f}"
                ax.text(
                    5,
                    9.7,
                    title,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    path_effects=[
                        plt.matplotlib.patheffects.withStroke(
                            linewidth=2, foreground="black"
                        )
                    ],
                )
                save_network_plot(fig, filepath)
                print(f"  ✅ {filename}")
            except Exception as e:
                print(f"  ❌ Error saving {filename}: {e}")
