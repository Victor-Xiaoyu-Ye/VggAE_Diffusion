#!/bin/bash
# Bounded cleanup for helper processes launched as jobs by the current shell.
# A PID from a completed job may have been reused; never signal kill -0 alone.
job_child_running() {
  local target=$1 active_pid
  for active_pid in $(jobs -pr); do
    [[ "${active_pid}" != "${target}" ]] || return 0
  done
  return 1
}

stop_job_child() {
  local target=${1:-} label=${2:-helper} grace=${SCENE_HELPER_STOP_TIMEOUT:-30} deadline
  [[ -n "${target}" ]] || return 0
  [[ "${grace}" =~ ^[1-9][0-9]*$ ]] || grace=30
  if job_child_running "${target}"; then
    echo "[launcher cleanup] stopping ${label} pid=${target}"
    kill -TERM "${target}" 2>/dev/null || true
    deadline=$((SECONDS+grace))
    while job_child_running "${target}" && (( SECONDS < deadline )); do sleep 0.2; done
    if job_child_running "${target}"; then
      echo "[launcher cleanup] ${label} exceeded ${grace}s; sending SIGKILL" >&2
      kill -KILL "${target}" 2>/dev/null || true
      deadline=$((SECONDS+3))
      while job_child_running "${target}" && (( SECONDS < deadline )); do sleep 0.2; done
    fi
    if job_child_running "${target}"; then
      echo "[launcher cleanup] ${label} is still blocked after SIGKILL; not waiting indefinitely" >&2
      return 1
    fi
  fi
  wait "${target}" 2>/dev/null || true
}
