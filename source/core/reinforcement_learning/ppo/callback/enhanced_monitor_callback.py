# enhanced_monitor_callback.py
from __future__ import annotations
import os
from collections import deque
from typing import Optional
import time
import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EnhancedMonitorAndCheckpointCallback(BaseCallback):
    """
    MonitorAndTimedCheckpointCallback과 동일한 기능에 추가로 logger를 통한 성능 로깅 기능을 제공합니다.

    VecMonitor의 episode 요약을 이용해 최근 n개 에피소드 누적보상 평균을 텐서보드에 기록하고,
    일정 시간 간격마다 최신 체크포인트로 모델을 덮어쓰며, logger를 통해 현재 성능을 출력합니다.

    Args:
        logger (logging.Logger): 성능 정보를 출력할 logger 객체
        rolling_n (int): 최근 n개 에피소드 리턴을 평균 내어 기록.
        rolling_tag (str): 텐서보드에 기록할 태그 이름.
        save_interval_sec (Optional[int]): 체크포인트 저장 간격(초). None이면 저장 비활성화.
        save_dir (Optional[str]): 체크포인트 저장 디렉토리 경로. None이면 저장 비활성화.
        save_sub_dir (str): save_dir 하위 서브 디렉토리 이름.
        save_basename (str): 저장 파일 기본 이름(확장자는 SB3가 .zip으로 자동 추가).
        save_on_train_end (bool): 학습 종료 시 마지막으로 한 번 저장할지 여부.
        log_interval_sec (Optional[int]): 성능 로깅 간격(초). None이면 로깅 비활성화.
        verbose (int): 출력 수준.
    """

    def __init__(
        self,
        logger: logging.Logger,
        rolling_n: int = 128,
        rolling_tag: str = "metrics/rolling_ep_return_from_monitor",
        save_interval_sec: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_sub_dir: str = "checkpoints",
        save_basename: str = "latest_model",
        save_on_train_end: bool = True,
        log_interval_sec: Optional[int] = 300,  # 5분마다 로깅
        verbose: int = 0,
    ):
        super().__init__(verbose)

        # Custom logger
        self.custom_logger = logger

        # logging
        self.rolling_n = int(rolling_n)
        self.rolling_tag = str(rolling_tag)
        self._recent: Optional[deque] = None
        self._recent_lengths: Optional[deque] = None

        # checkpoint
        self.save_interval_sec = (
            int(save_interval_sec) if save_interval_sec is not None else None
        )
        self.save_dir = save_dir
        self.save_sub_dir = str(save_sub_dir)
        self.save_basename = str(save_basename)
        self.save_on_train_end = bool(save_on_train_end)
        self._save_dir: Optional[str] = None
        self._save_path: Optional[str] = None
        self._last_save_time: Optional[float] = None

        # performance logging
        self.log_interval_sec = (
            int(log_interval_sec) if log_interval_sec is not None else None
        )
        self._last_log_time: Optional[float] = None
        self._total_episodes = 0
        self._start_time: Optional[float] = None

    # ----------------------- SB3 lifecycle hooks -----------------------

    def _on_training_start(self) -> None:
        self._recent = deque(maxlen=self.rolling_n)
        self._recent_lengths = deque(maxlen=self.rolling_n)
        self._start_time = time.time()

        # 학습 시작 로깅
        self.custom_logger.info("=" * 60)
        self.custom_logger.info("🚀 PPO 학습 시작!")
        self.custom_logger.info(f"Rolling window size: {self.rolling_n} episodes")
        if self.log_interval_sec:
            self.custom_logger.info(f"성능 로깅 간격: {self.log_interval_sec}초")
        self.custom_logger.info("=" * 60)

        if self.save_interval_sec is not None and self.save_dir is not None:
            self._save_dir = os.path.join(self.save_dir, self.save_sub_dir)
            os.makedirs(self._save_dir, exist_ok=True)
            self._save_path = os.path.join(self._save_dir, self.save_basename)
            self._last_save_time = time.time()
            if self.verbose:
                print(
                    f"[Callback] Checkpoints -> {self._save_dir}/{self.save_basename}_{{step}}_reward_{{reward}}.zip (every {self.save_interval_sec}s)"
                )

        if self.log_interval_sec is not None:
            self._last_log_time = time.time()

    def _on_step(self) -> bool:
        # --------- 1) Rolling episode return from VecMonitor ---------
        infos = self.locals.get("infos")  # list[dict], len = num_envs
        dones = self.locals.get("dones")  # np.ndarray(bool) or list

        if infos is not None and dones is not None:
            dones = np.asarray(dones)
            if np.any(dones):
                finished_idx = np.nonzero(dones)[0]
                for i in finished_idx:
                    ep = infos[i].get("episode", None)
                    if ep is None:
                        continue
                    # ep: {"r": return, "l": length, "t": seconds}
                    ep_r = float(ep.get("r", 0.0))
                    ep_l = float(ep.get("l", 0.0))

                    self._recent.append(ep_r)
                    self._recent_lengths.append(ep_l)
                    self._total_episodes += 1

                if self._recent and len(self._recent) > 0:
                    mean_ep_r = float(np.mean(self._recent))
                    self.logger.record(self.rolling_tag, mean_ep_r)

        # --------- 2) Time-based performance logging ---------
        if self.log_interval_sec is not None and self._last_log_time is not None:
            now = time.time()
            if now - self._last_log_time >= self.log_interval_sec:
                self._log_performance()
                self._last_log_time = now

        # --------- 3) Time-based latest checkpoint overwrite ---------
        if (
            self.save_interval_sec is not None
            and self._save_path is not None
            and self.save_dir is not None
        ):
            now = time.time()
            if (self._last_save_time is None) or (
                now - self._last_save_time >= self.save_interval_sec
            ):
                self._save_latest()
                self._last_save_time = now

        return True

    def _on_training_end(self) -> None:
        # 최종 성능 로깅
        self._log_performance(is_final=True)

        if (
            self.save_on_train_end
            and self.save_interval_sec is not None
            and self._save_path is not None
            and self.save_dir is not None
        ):
            self._save_latest(suffix="_final")
            if self.verbose:
                print(f"[Callback] Final checkpoint saved at end of training.")

    # ----------------------- helpers -----------------------

    def _log_performance(self, is_final: bool = False) -> None:
        """현재 성능 정보를 logger를 통해 출력합니다."""
        if not self._recent or len(self._recent) == 0:
            return

        current_time = time.time()
        elapsed_time = current_time - self._start_time if self._start_time else 0

        # 기본 통계
        mean_reward = float(np.mean(self._recent))
        std_reward = float(np.std(self._recent)) if len(self._recent) > 1 else 0.0
        min_reward = float(np.min(self._recent))
        max_reward = float(np.max(self._recent))

        mean_length = (
            float(np.mean(self._recent_lengths)) if self._recent_lengths else 0.0
        )

        # 학습 진행률
        fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        episodes_per_hour = (
            self._total_episodes / (elapsed_time / 3600) if elapsed_time > 0 else 0
        )

        # 로그 메시지 구성
        prefix = "🎯 최종 성능 요약" if is_final else "📊 현재 성능 상태"

        self.custom_logger.info("-" * 50)
        self.custom_logger.info(f"{prefix} (최근 {len(self._recent)}개 에피소드)")
        self.custom_logger.info(
            f"⏰ 경과시간: {elapsed_time/3600:.1f}시간 ({elapsed_time/60:.1f}분)"
        )
        self.custom_logger.info(f"🎮 총 스텝: {self.num_timesteps:,} | FPS: {fps:.1f}")
        self.custom_logger.info(
            f"🏆 에피소드: {self._total_episodes}개 | 시간당: {episodes_per_hour:.1f}개"
        )
        self.custom_logger.info(f"💰 보상 평균: {mean_reward:.2f} ± {std_reward:.2f}")
        self.custom_logger.info(f"📈 보상 범위: [{min_reward:.2f}, {max_reward:.2f}]")
        self.custom_logger.info(f"📏 평균 길이: {mean_length:.1f} 스텝")

        # 성능 트렌드 (최근 절반과 비교)
        if len(self._recent) >= 20:
            recent_half = list(self._recent)[len(self._recent) // 2 :]
            recent_half_mean = float(np.mean(recent_half))
            trend = (
                "📈 상승"
                if recent_half_mean > mean_reward
                else "📉 하락" if recent_half_mean < mean_reward else "➡️ 안정"
            )
            self.custom_logger.info(
                f"🔄 트렌드: {trend} (최근 절반 평균: {recent_half_mean:.2f})"
            )

        self.custom_logger.info("-" * 50)

    def _save_latest(self, suffix: str = "") -> None:
        """
        최신 체크포인트를 보상과 스텝 정보가 포함된 파일명으로 저장.
        파일명 형식: {basename}_{step}_reward_{reward}.pth
        suffix를 주면 파일명 뒤에 덧붙여 저장(예: latest_model_final.zip)
        """
        # 현재 평균 보상 계산 (3자리 소수점으로 포맷)
        current_reward = 0.0
        if self._recent and len(self._recent) > 0:
            current_reward = float(np.mean(self._recent))

        # 파일명 생성: basename_step_reward_X.XXX.pth
        reward_str = f"{current_reward:06.3f}".replace(".", "_")  # 0.090 -> 0_090
        filename = f"{self.save_basename}_{self.num_timesteps}_reward_{reward_str}.pth"

        if suffix:
            # suffix가 있으면 .pth 앞에 삽입
            filename = filename.replace(".pth", f"{suffix}.pth")

        # 전체 저장 경로
        full_path = (
            os.path.join(self._save_dir, filename) if self._save_dir else filename
        )

        # 모델 저장 (SB3는 자동으로 .zip을 추가하므로 .pth 제거)
        save_path = full_path.replace(".pth", "")
        self.model.save(save_path)

        # 체크포인트 저장도 logger로 기록
        save_msg = f"💾 체크포인트 저장: {save_path}.zip"
        save_msg += (
            f" | 스텝: {self.num_timesteps:,} | 에피소드: {self._total_episodes}"
        )
        save_msg += f" | 평균보상: {current_reward:.3f}"

        self.custom_logger.info(save_msg)

        if self.verbose:
            print(
                f"[Checkpoint] Saved -> {save_path}.zip @ step={self.num_timesteps}, reward={current_reward:.3f}"
            )
