"""测试用的 MCP 服务器（stdio）：工具列表分两页；echo 调用前先反过来 ping 一下客户端；
还有报错、带图、崩溃、很慢的工具。往 stderr 打日志（客户端要读掉，不然会卡住）。"""

import json
import sys
import time

TOOLS = [
    {"name": "echo", "description": "Echo the text back. " + "x" * 3000,
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "times": {"type": "integer",
                                                                                            "minimum": 1}},
                     "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "picture", "description": "Returns an image.", "inputSchema": {"type": "object"}},
    {"name": "crash", "description": "Exits the server.", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Sleeps.", "inputSchema": {"type": "object"}},
]
_pending_ping = None


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    print(f"got {msg.get('method') or 'response'}", file=sys.stderr, flush=True)
    method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method is None:                                  # 客户端对我们 ping 的响应：现在回 echo 的结果
        if _pending_ping and msg.get("id") == "ping-1" and "result" in msg:
            call_id, args = _pending_ping
            text = args["text"] * args.get("times", 1)
            send({"jsonrpc": "2.0", "id": call_id, "result": {"content": [{"type": "text", "text": text}]}})
        continue
    if rid is None:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake"}, "instructions": "Use echo to repeat text."}})
    elif method == "tools/list":
        page = 1 if params.get("cursor") == "p2" else 0
        result = {"tools": TOOLS[:2]} if page == 0 else {"tools": TOOLS[2:]}
        if page == 0:
            result["nextCursor"] = "p2"
        send({"jsonrpc": "2.0", "id": rid, "result": result})
    elif method == "tools/call":
        name, args = params["name"], params.get("arguments") or {}
        if name == "echo":
            _pending_ping = (rid, args)
            send({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "boom"}],
                                                          "isError": True}})
        elif name == "picture":
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [
                {"type": "text", "text": "a chart"}, {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}]}})
        elif name == "crash":
            print("fatal: crashing", file=sys.stderr, flush=True)
            sys.exit(1)
        elif name == "slow":
            time.sleep(5)
    else:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})
