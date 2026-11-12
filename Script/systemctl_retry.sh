#!/usr/bin/env bash

# Package upgrades can reexecute systemd while a client waits for a start job.
# Retrying start joins the existing job; it does not restart a healthy service.
systemctl_retry() {
    local attempt output status
    for attempt in 1 2 3 4 5 6; do
        if output="$(LC_ALL=C systemctl "$@" 2>&1)"; then
            [[ -z "$output" ]] || printf '%s\n' "$output"
            return 0
        else
            status=$?
        fi
        printf '%s\n' "$output" >&2
        case "$output" in
            *"D-Bus connection terminated"*|*"Connection reset by peer"*|*"Failed to connect to bus"*|*"Transport endpoint is not connected"*)
                if (( attempt < 6 )); then
                    printf 'Systemd connection interrupted; retrying in 5 seconds (%s/6).\n' "$attempt" >&2
                    sleep 5
                    continue
                fi
                ;;
        esac
        return "$status"
    done
}
