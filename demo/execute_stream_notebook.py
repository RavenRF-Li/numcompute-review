"""Execute the streaming notebook with a Jupyter kernel and save its outputs."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "demo" / "stream_demo.ipynb"


def _execute_cell(client, cell):
    request = client.execute("".join(cell["source"]), allow_stdin=False)
    outputs = []
    while True:
        message = client.get_iopub_msg(timeout=60)
        if message["parent_header"].get("msg_id") != request:
            continue
        kind, content = message["msg_type"], message["content"]
        if kind == "error":
            raise RuntimeError("\n".join(content["traceback"]))
        if kind in ("stream", "display_data", "execute_result"):
            fields = {
                "stream": ("name", "text"),
                "display_data": ("data", "metadata"),
                "execute_result": ("data", "metadata", "execution_count"),
            }[kind]
            outputs.append({"output_type": kind, **{name: content[name] for name in fields}})
        if kind == "status" and content["execution_state"] == "idle":
            break

    reply = client.get_shell_msg(timeout=60)
    while reply["parent_header"].get("msg_id") != request:
        reply = client.get_shell_msg(timeout=60)
    if reply["content"]["status"] != "ok":
        raise RuntimeError(reply["content"])
    cell["execution_count"] = reply["content"]["execution_count"]
    cell["outputs"] = outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="aiml")
    args = parser.parse_args()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    with TemporaryDirectory(prefix="numcompute-nb-", dir="/tmp") as directory:
        temporary = Path(directory)
        manager = KernelManager(kernel_name=args.kernel, transport="ipc", ip=str(temporary / "k"))
        env = {
            **os.environ,
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
            "IPYTHONDIR": str(temporary / "ipython"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        manager.start_kernel(cwd=ROOT, env=env)
        client = manager.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
            for cell in cells:
                _execute_cell(client, cell)
        finally:
            client.stop_channels()
            manager.shutdown_kernel(now=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Executed {len(cells)} code cells on the {args.kernel!r} Jupyter kernel and saved {NOTEBOOK}")


if __name__ == "__main__":
    main()
