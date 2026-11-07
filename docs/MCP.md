# Retake over MCP

`retake_mcp.py` exposes Retake to any MCP client. The split is deliberate. The client does what Retake can't, which is
understand what you meant. Retake does what a language model shouldn't be trusted with, which is deciding which real
audio a phrase refers to. Ask for "the bit where I fumbled the sponsor read" and Retake answers with exact word tokens
and timestamps, or says it couldn't find them. A client can never name a raw time range to delete, and it can't invent
a token ID.

## Connecting

Start Retake, then add it to your client's MCP config (Claude Desktop, Claude Code, or anything else):

```json
{
  "mcpServers": {
    "retake": {
      "command": "python",
      "args": ["/path/to/Retake/retake_mcp.py"]
    }
  }
}
```

Clients that speak streamable HTTP can use `http://localhost:8710/mcp`, which the running editor serves itself.
`retake_mcp.py --http --port 8711` runs the same tools as a standalone HTTP server. Set `RETAKE_URL` if Retake is
listening somewhere other than `http://127.0.0.1:8710`.

`mcp` is in `requirements.txt` but the editor doesn't need it. Without it, Retake runs as usual with no `/mcp`
endpoint.

## Safety

Every tool that changes something previews first. `apply_cuts` reports the token IDs that would change and the
resulting duration, and writes nothing until you call it again with `dry_run=false`. Applied edits go onto an undo
stack of project snapshots that `undo` unwinds one step at a time. Edits are deltas, so cutting five words never
resets the rest of your work.

## Tools

| Group | Tools |
|---|---|
| Reading | `get_status`, `list_projects`, `open_project`, `get_edit_summary`, `read_transcript`, `find_phrase`, `list_retakes` |
| Editing | `plan_edit`, `get_proposals`, `apply_cuts`, `apply_proposals`, `undo`, `set_markers`, `set_voice_enhancement` |
| Jobs | `start_export`, `get_job_status`, `export_text`, `get_latest_export`, `recalibrate_timing` |

## The retake-brain skill

`retake-brain.skill` is a Claude skill that turns a raw recording into a cut list. It works out what the video is
trying to say, groups every attempt at each line, and keeps one per group: the last complete attempt wins. It only
flags the calls that need ears, like a splice in the middle of a sentence, a delivery choice, or a fact the speaker
contradicted.

With the MCP server connected, it reads the transcript from the open project and applies the list itself, through a
dry run you approve. Without it, it works from an SRT you upload and prints the list for you to paste into **Cut by
instruction**, in the same command grammar Retake's parser reads.

Install the `.skill` file into Claude. To change it, edit `skills/retake-brain/SKILL.md` and rebuild:

```bash
python skills/build_skill.py
```

The tests run the skill's own worked example through the real parser, so the two can't drift apart.
