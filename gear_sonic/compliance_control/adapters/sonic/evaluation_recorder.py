"""IsaacLab lifecycle bridge for the SONIC aligned-trace collector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
import numpy as np
import torch

from .evaluation import (
    SonicEvaluationProtocol,
    SonicEvaluationTraceCollector,
    apply_sonic_evaluation_protocol,
    snapshot_from_sonic_commands,
)

if TYPE_CHECKING:
    from isaaclab import envs


@configclass
class SonicEvaluationTraceRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for one in-memory, bounded Phase-6 trace recorder."""

    class_type: type = None
    trial_name: str = ""
    seed_id: int = 0
    tracking_command_name: str = "motion"
    compliance_command_name: str = "motion_compliance"
    protocol_enabled: bool = False
    protocol_operational_enabled: bool = False
    active_site_ids: tuple[str, ...] = ()
    force_threshold_n: float = 10.0
    reference_offset_common_m: tuple[float, float, float] = (0.05, 0.0, 0.0)
    max_rows: int = 100_000


@configclass
class SonicEvaluationRecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorder manager that keeps IsaacLab's HDF5 exporter fully disabled."""

    dataset_export_mode = recorder_manager.DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    export_in_close = False
    motion_compliance_trace = None


class SonicEvaluationTraceRecorderTerm(recorder_manager.RecorderTerm):
    """Capture terminal physics before reset and the cleared post-reset state."""

    cfg: SonicEvaluationTraceRecorderCfg

    def __init__(self, cfg: SonicEvaluationTraceRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env
        self.protocol = SonicEvaluationProtocol(
            enabled=cfg.protocol_enabled,
            operational_enabled=cfg.protocol_operational_enabled,
            active_site_ids=tuple(cfg.active_site_ids),
            force_threshold_n=cfg.force_threshold_n,
            reference_offset_common_m=tuple(cfg.reference_offset_common_m),
        )
        self.collector: SonicEvaluationTraceCollector | None = None

    def _command(self):
        command = self.env.command_manager.get_term(self.cfg.compliance_command_name)
        tracking = command._tracking_term()
        if tracking is not self.env.command_manager.get_term(self.cfg.tracking_command_name):
            raise RuntimeError("compliance command is bound to a different tracking command")
        return command

    def _snapshot_and_collector(self):
        snapshot = snapshot_from_sonic_commands(self._command())
        if self.collector is None:
            self.collector = SonicEvaluationTraceCollector(
                trial_name=self.cfg.trial_name,
                seed_id=self.cfg.seed_id,
                step_dt_s=float(self.env.step_dt),
                site_ids=snapshot.site_ids,
                point_ids=snapshot.point_ids,
                max_rows=self.cfg.max_rows,
            )
        return snapshot, self.collector

    @staticmethod
    def _env_id_tuple(env_ids: Sequence[int] | torch.Tensor | None) -> tuple[int, ...] | None:
        if env_ids is None:
            return None
        if isinstance(env_ids, torch.Tensor):
            return tuple(int(value) for value in env_ids.detach().cpu().tolist())
        return tuple(int(value) for value in env_ids)

    def record_post_reset(
        self,
        env_ids: Sequence[int] | None,
    ) -> tuple[str | None, torch.Tensor | dict | None]:
        """Apply the protocol before observations, then capture cleared force."""

        command = self._command()
        apply_sonic_evaluation_protocol(command, self.protocol, env_ids)
        snapshot, collector = self._snapshot_and_collector()
        collector.record_post_reset(snapshot, self._env_id_tuple(env_ids))
        return None, None

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Capture the just-finished physics state before automatic reset."""

        snapshot, collector = self._snapshot_and_collector()
        terminal = self.env.reset_buf.detach().to(device="cpu", dtype=torch.bool).numpy()
        terminated = (
            self.env.reset_terminated.detach().to(device="cpu", dtype=torch.bool).numpy()
        )
        timed_out = self.env.reset_time_outs.detach().to(device="cpu", dtype=torch.bool).numpy()
        fall = np.asarray(terminated, dtype=np.bool_)
        success = np.asarray(timed_out & ~terminated, dtype=np.bool_)
        collector.record_post_step(
            snapshot,
            terminal_mask=np.asarray(terminal, dtype=np.bool_),
            success_mask=success,
            fall_mask=fall,
        )
        return None, None

    def finalize_trace(
        self,
        *,
        natural_timeout_env_ids: Sequence[int],
        failed_env_ids: Sequence[int],
    ):
        """Return a trace only after the observed natural motion timeout."""

        if self.collector is None:
            raise RuntimeError("recorder did not observe a simulator reset")
        return self.collector.finalize(
            natural_timeout_env_ids=natural_timeout_env_ids,
            failed_env_ids=failed_env_ids,
        )

    def adapter_evidence_report(self) -> dict[str, object]:
        if self.collector is None:
            raise RuntimeError("recorder did not observe a simulator reset")
        return self.collector.adapter_evidence_report()


def make_sonic_evaluation_recorders_cfg(
    *,
    trial_name: str,
    seed_id: int,
    protocol: SonicEvaluationProtocol,
    max_rows: int,
) -> SonicEvaluationRecordersCfg:
    """Build the opt-in recorder config without modifying general env config."""

    term_cfg = SonicEvaluationTraceRecorderCfg(
        trial_name=trial_name,
        seed_id=seed_id,
        protocol_enabled=protocol.enabled,
        protocol_operational_enabled=protocol.operational_enabled,
        active_site_ids=protocol.active_site_ids,
        force_threshold_n=protocol.force_threshold_n,
        reference_offset_common_m=protocol.reference_offset_common_m,
        max_rows=max_rows,
    )
    term_cfg.class_type = SonicEvaluationTraceRecorderTerm
    cfg = SonicEvaluationRecordersCfg()
    cfg.motion_compliance_trace = term_cfg
    return cfg
