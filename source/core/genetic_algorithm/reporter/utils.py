import os
import neat
import matplotlib

matplotlib.use("Agg")  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from matplotlib.patches import FancyBboxPatch


def save_plot_stats(statistics, save_dir: str, filename: str = "fitness_stats.png"):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, "b-", label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, "g-.", label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def save_plot_species(statistics, save_dir: str, filename: str = "species_stats.png"):
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T
    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# ----------------------------------------------


def create_network_plot(genome, config, node_names=None):
    """
    NEAT 네트워크 구조를 matplotlib figure로 생성합니다.

    Args:
        genome: NEAT genome 객체
        config: NEAT config 객체
        node_names: 노드 이름 딕셔너리 (선택사항)

    Returns:
        tuple: (fig, ax) matplotlib figure와 axes 객체
    """

    if node_names is None:
        node_names = {}

    # 그래프 설정
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # 노드 위치 계산
    input_nodes = [i for i in range(-config.genome_config.num_inputs, 0)]
    output_nodes = [i for i in range(config.genome_config.num_outputs)]
    hidden_nodes = [
        i for i in genome.nodes.keys() if i not in input_nodes and i not in output_nodes
    ]

    # 노드 위치 설정
    node_positions = {}

    # 입력 노드 위치 (왼쪽)
    if input_nodes:
        for i, node_id in enumerate(input_nodes):
            y_pos = (
                9 - (i * 8 / max(1, len(input_nodes) - 1))
                if len(input_nodes) > 1
                else 5
            )
            node_positions[node_id] = (1, y_pos)

    # 출력 노드 위치 (오른쪽)
    if output_nodes:
        for i, node_id in enumerate(output_nodes):
            y_pos = (
                9 - (i * 8 / max(1, len(output_nodes) - 1))
                if len(output_nodes) > 1
                else 5
            )
            node_positions[node_id] = (9, y_pos)

    # 히든 노드 위치 (중간)
    if hidden_nodes:
        for i, node_id in enumerate(hidden_nodes):
            x_pos = 3 + (i % 3) * 2  # 여러 열로 배치
            y_pos = 8 - (i // 3) * 2  # 여러 행으로 배치
            node_positions[node_id] = (x_pos, y_pos)

    # 연결선 그리기
    for (input_id, output_id), conn in genome.connections.items():
        if conn.enabled and input_id in node_positions and output_id in node_positions:
            x1, y1 = node_positions[input_id]
            x2, y2 = node_positions[output_id]

            # 연결선 색상 (가중치에 따라)
            color = "green" if conn.weight > 0 else "red"
            alpha = min(abs(conn.weight) / 5.0, 1.0)  # 가중치 크기에 따른 투명도
            linewidth = max(abs(conn.weight) / 2.0, 0.5)

            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)

            # 화살표 추가
            dx, dy = x2 - x1, y2 - y1
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color=color, alpha=alpha, lw=linewidth
                ),
            )

    # 노드 그리기
    for node_id, (x, y) in node_positions.items():
        # 노드 타입에 따른 색상
        if node_id in input_nodes:
            color = "lightblue"
            shape = "s"  # 사각형
        elif node_id in output_nodes:
            color = "lightcoral"
            shape = "o"  # 원형
        else:
            color = "lightgreen"
            shape = "o"  # 원형

        # 노드 그리기
        if shape == "s":
            # 사각형 노드
            rect = FancyBboxPatch(
                (x - 0.3, y - 0.3),
                0.6,
                0.6,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)
        else:
            # 원형 노드
            circle = plt.Circle((x, y), 0.3, color=color, ec="black", linewidth=2)
            ax.add_patch(circle)

        # 노드 라벨 (흰색 텍스트에 검은색 테두리)
        label = node_names.get(node_id, str(node_id))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            path_effects=[
                plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
            ],
        )

        # 바이어스 정보 (히든/출력 노드만)
        if node_id in genome.nodes:
            bias = genome.nodes[node_id].bias
            ax.text(
                x,
                y - 0.5,
                f"{bias:.1f}",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(
                        linewidth=1, foreground="black"
                    )
                ],
            )

    # # 제목 (간단하고 작게)
    # ax.text(
    #     5,
    #     9.7,
    #     "NEAT Network",
    #     ha="center",
    #     va="center",
    #     fontsize=12,
    #     fontweight="bold",
    #     color="white",
    #     path_effects=[
    #         plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
    #     ],
    # )

    return fig, ax


def save_network_plot(fig, filename="network_structure.png"):
    """
    matplotlib figure를 파일로 저장합니다.

    Args:
        fig: matplotlib figure 객체
        filename: 저장할 파일명
    """
    # 화면에 표시하지 않고 직접 저장
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Network structure saved to {filename}")
    plt.close(fig)


def draw_network_structure(
    genome, config, filename="network_structure.png", node_names=None
):
    """
    NEAT 네트워크 구조를 그리고 파일로 저장합니다. (기존 API 호환성 유지)

    Args:
        genome: NEAT genome 객체
        config: NEAT config 객체
        filename: 저장할 파일명
        node_names: 노드 이름 딕셔너리 (선택사항)
    """
    fig, ax = create_network_plot(genome, config, node_names)
    save_network_plot(fig, filename)


class SpeciesTracker(neat.reporting.BaseReporter):
    """각 종의 최고 성능 genome을 추적하고 업데이트하는 리포터"""

    def __init__(self, config):
        self.config = config
        self.generation = 0

        # 각 종의 최고 성능 기록 {species_id: {'fitness': float, 'genome': genome, 'generation': int}}
        self.species_best = {}

        # LunarLander 노드 이름 설정
        self.node_names = {
            -1: "X Position",
            -2: "Y Position",
            -3: "X Velocity",
            -4: "Y Velocity",
            -5: "Angle",
            -6: "Angular Velocity",
            -7: "Left Leg Contact",
            -8: "Right Leg Contact",
            0: "No Action",
            1: "Left Engine",
            2: "Main Engine",
            3: "Right Engine",
        }

    def start_generation(self, generation):
        """세대 시작 시 호출"""
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """각 세대 평가 후 모든 종의 최고 성능 확인 및 업데이트"""

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
                print(
                    f"\n🎉 NEW SPECIES DETECTED! Species ID: {species_id} (Generation {self.generation})"
                )
                update_needed = True
            elif current_fitness > self.species_best[species_id]["fitness"]:
                print(
                    f"\n📈 SPECIES {species_id} IMPROVED! New best fitness: {current_fitness:.2f} (Generation {self.generation})"
                )
                print(
                    f"   Previous best: {self.species_best[species_id]['fitness']:.2f} (Generation {self.species_best[species_id]['generation']})"
                )
                update_needed = True

            if update_needed:
                # 기록 업데이트
                self.species_best[species_id] = {
                    "fitness": current_fitness,
                    "genome": current_best,
                    "generation": self.generation,
                }

                # 파일명 생성 (항상 같은 이름으로 덮어쓰기)
                filename = f"species_{species_id}_best_structure.png"

                # 네트워크 구조 저장
                try:
                    draw_network_structure(
                        current_best, self.config, filename, self.node_names
                    )

                    if is_new_species:
                        print(f"✅ Species {species_id} structure saved to {filename}")
                    else:
                        print(
                            f"🔄 Species {species_id} structure updated in {filename}"
                        )

                    # 상세 정보 출력
                    print(f"   - Fitness: {current_fitness:.2f}")
                    print(f"   - Generation: {self.generation}")
                    print(f"   - Nodes: {len(current_best.nodes)}")
                    print(
                        f"   - Active Connections: {len([c for c in current_best.connections.values() if c.enabled])}"
                    )

                except Exception as e:
                    print(f"❌ Error saving species {species_id} structure: {e}")

        # 현재 상태 요약 출력
        if self.generation % 10 == 0:  # 10세대마다 요약 출력
            print(f"\n📊 SPECIES SUMMARY (Generation {self.generation}):")
            for species_id, record in self.species_best.items():
                print(
                    f"   Species {species_id}: Best={record['fitness']:.2f} (Gen {record['generation']})"
                )
