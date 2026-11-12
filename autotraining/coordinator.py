"""Lambda dispatch for the nightly Step Functions state machine and cost guard."""

import execution
import transfer


def handler(event, context):
    # Terminal workflow events also cover manual abort and global timeouts.
    if event.get("source") == "aws.states":
        detail = event["detail"]
        if detail["status"] in ("FAILED", "TIMED_OUT", "ABORTED"):
            return execution.cleanup(
                detail["executionArn"], {"Error": detail["status"]}
            )
        return {"ignored": True}
    action = event["action"]
    if action == "watchdog":
        return execution.watchdog()
    if action == "snapshot":
        return transfer.start_snapshot()
    if action == "prepare":
        return transfer.prepare(
            event["command_id"], event["execution_id"], event.get("request")
        )
    if action == "publish":
        return execution.publish(event["job"])
    if action == "publication_status":
        return execution.publication_status(event["job"], event["command_id"])
    if action == "start":
        return execution.start(event["job"])
    if action == "launch":
        return execution.launch(event["job"], event.get("attempt", 0))
    if action == "poll":
        execution.require_owner(event["job"])
        return execution.poll(event["command_id"])
    if action == "complete":
        return execution.finish(event["job"])
    if action in ("stop", "fail"):
        return execution.cleanup(event["execution_id"], event.get("error"))
    raise ValueError("Unknown action")
