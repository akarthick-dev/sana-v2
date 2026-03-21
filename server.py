import json
from pathlib import Path

from aiohttp import web


BASE_DIR = Path(__file__).resolve().parent
connected_clients: set[web.WebSocketResponse] = set()


async def index(_request: web.Request) -> web.FileResponse:
	return web.FileResponse(BASE_DIR / "index.html")


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
	ws = web.WebSocketResponse(heartbeat=20)
	await ws.prepare(request)

	connected_clients.add(ws)
	await ws.send_json({"type": "status", "message": "WebSocket connected"})

	try:
		async for _msg in ws:
			# Browser -> server messages are currently ignored.
			pass
	finally:
		connected_clients.discard(ws)

	return ws


async def broadcast(payload: dict) -> int:
	dead_clients = []
	delivered = 0

	for ws in connected_clients:
		if ws.closed:
			dead_clients.append(ws)
			continue

		try:
			await ws.send_json(payload)
			delivered += 1
		except Exception:
			dead_clients.append(ws)

	for ws in dead_clients:
		connected_clients.discard(ws)

	return delivered


async def emit_handler(request: web.Request) -> web.Response:
	try:
		data = await request.json()
	except json.JSONDecodeError:
		return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

	role = str(data.get("role", "assistant")).strip().lower()
	text = str(data.get("text", "")).strip()
	audio_url = data.get("audio_url")
	event = str(data.get("event", "chat")).strip().lower()

	if role not in {"user", "assistant", "system"}:
		return web.json_response({"ok": False, "error": "Invalid role"}, status=400)

	if not text:
		return web.json_response({"ok": False, "error": "Text is required"}, status=400)

	if event not in {"chat", "audio"}:
		return web.json_response({"ok": False, "error": "Invalid event"}, status=400)

	payload = {"type": "chat", "role": role, "text": text, "event": event}
	if isinstance(audio_url, str) and audio_url.strip():
		payload["audio_url"] = audio_url.strip()
	delivered = await broadcast(payload)
	return web.json_response({"ok": True, "clients": delivered})


def create_app() -> web.Application:
	app = web.Application()
	app.router.add_get("/", index)
	app.router.add_get("/ws", websocket_handler)
	app.router.add_post("/emit", emit_handler)

	# Serve local assets used by index.html.
	app.router.add_static("/", BASE_DIR)
	return app


if __name__ == "__main__":
	web.run_app(create_app(), host="127.0.0.1", port=8000)
