"""Checkpoint and staged-finetuning workflow for motion compliance."""

from .actor import MOTION_COMPLIANCE_ACTOR_TARGET, MotionComplianceFrozenNoiseActor
from .audit import (
    TrainedCheckpointAuditReport,
    audit_motion_compliance_exposure_report,
    audit_trained_motion_compliance_checkpoint,
)
from .checkpoint import (
    MOTION_COMPLIANCE_MIGRATION_KEY,
    OFFICIAL_SONIC_RELEASE_SHA256,
    CheckpointMigrationReport,
    LoadStateReport,
    ResumeStatePayload,
    audit_migrated_init_checkpoint,
    critic_added_columns,
    migrate_motion_compliance_checkpoint,
    migrate_motion_compliance_checkpoint_file,
    migrate_official_sonic_release_checkpoint_file,
    strict_load_policy_value_state,
    validate_checkpoint_sha256,
    validate_strict_resume_payload,
)
from .callback import MotionComplianceExposureCallback
from .finetune import (
    FinetuneStageReport,
    configure_motion_compliance_finetune_stage,
    validate_motion_compliance_workflow_config,
    validate_optimizer_parameter_set,
)
from .paths import (
    MOTION_COMPLIANCE_RUNS_ROOT,
    OFFICIAL_SAMPLE_ROBOT_MOTION,
    OFFICIAL_SAMPLE_SMPL_MOTION,
    OFFICIAL_SONIC_RELEASE_CHECKPOINT,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)
from .residual_policy import (
    CONDITION_OBSERVATION_KEY,
    MOTION_COMPLIANCE_BACKBONE_TARGET,
    MOTION_COMPLIANCE_CRITIC_TARGET,
    PRIVILEGED_OBSERVATION_KEY,
    MotionComplianceResidualCritic,
    MotionComplianceUniversalTokenModule,
    ZeroInitializedResidualMLP,
    motion_compliance_residual_parameters,
)

__all__ = [
    "MOTION_COMPLIANCE_MIGRATION_KEY",
    "MOTION_COMPLIANCE_ACTOR_TARGET",
    "OFFICIAL_SONIC_RELEASE_SHA256",
    "CheckpointMigrationReport",
    "CONDITION_OBSERVATION_KEY",
    "FinetuneStageReport",
    "LoadStateReport",
    "MOTION_COMPLIANCE_RUNS_ROOT",
    "MOTION_COMPLIANCE_BACKBONE_TARGET",
    "MOTION_COMPLIANCE_CRITIC_TARGET",
    "MotionComplianceExposureCallback",
    "MotionComplianceFrozenNoiseActor",
    "MotionComplianceResidualCritic",
    "MotionComplianceUniversalTokenModule",
    "OFFICIAL_SAMPLE_ROBOT_MOTION",
    "OFFICIAL_SAMPLE_SMPL_MOTION",
    "OFFICIAL_SONIC_RELEASE_CHECKPOINT",
    "ResumeStatePayload",
    "PRIVILEGED_OBSERVATION_KEY",
    "TrainedCheckpointAuditReport",
    "audit_motion_compliance_exposure_report",
    "audit_migrated_init_checkpoint",
    "audit_trained_motion_compliance_checkpoint",
    "configure_motion_compliance_finetune_stage",
    "critic_added_columns",
    "migrate_motion_compliance_checkpoint",
    "migrate_motion_compliance_checkpoint_file",
    "migrate_official_sonic_release_checkpoint_file",
    "strict_load_policy_value_state",
    "validate_checkpoint_sha256",
    "validate_distinct_artifact_paths",
    "validate_motion_compliance_run_path",
    "validate_motion_compliance_workflow_config",
    "validate_optimizer_parameter_set",
    "validate_strict_resume_payload",
    "ZeroInitializedResidualMLP",
    "motion_compliance_residual_parameters",
]
