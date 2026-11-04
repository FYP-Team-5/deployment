"""Lambda dispatch for the nightly Step Functions state machine."""
import logging

import execution
import transfer

log = logging.getLogger(__name__)


def handler(event, context):
    action = event["action"]
    if action == "prepare":
        return transfer.prepare(context)
    if action == "start":
        return execution.start()
    if action == "launch":
        return execution.launch(event["job"], event.get("attempt", 0))
    if action == "poll":
        return execution.poll(event["command_id"])
    if action == "complete":
        return execution.finish(event["job"])
    if action == "stop":
        result = execution.stop()
        transfer.unlock(event["job"]["checksum"])
        return result
    if action == "fail":
        job = event.get("job") or {}
        log.error("Training workflow failed (version=%s)", job.get("version_id"))
        return execution.failure(job)
    raise ValueError("Unknown action")
