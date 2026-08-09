from failure_analysis.gemini_crog_evidence_v1.supervisor import should_restart


def test_supervisor_never_restarts_active_or_hard_stopped_runner():
    assert not should_restart(active=True, phase_status="running")
    assert not should_restart(active=False, phase_status="blocked_budget")
    assert not should_restart(active=False, phase_status="blocked_provider_drift")
    assert not should_restart(active=False, phase_status="complete")


def test_supervisor_restarts_only_nonterminal_inactive_state():
    assert should_restart(active=False, phase_status="running")
    assert should_restart(active=False, phase_status="failed")
