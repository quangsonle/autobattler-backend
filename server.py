import os
import time
import math
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration & Security ---
ROOM_PASSWORD = os.getenv("ROOM_PASSWORD", "arena123")  # Change your password here
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 900  # 15 minutes in seconds

MAP_WIDTH = 50.0
MAP_HEIGHT = 300.0
TPS = 30
PLAYER_SPEED = 25.0 / TPS
BULLET_SPEED = 50.0 / TPS
FIRE_COOLDOWN = int(0.5 * TPS)
HITBOX_RADIUS = 1.6

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
failed_attempts: Dict[str, Dict] = {}  # ip -> {"count": int, "locked_until": float}

class Player:
    def __init__(self, pid: str, username: str, x: float, y: float, y_min: float, y_max: float):
        self.pid = pid
        self.username = username
        self.x = x
        self.y = y
        self.y_min = y_min
        self.y_max = y_max
        self.score = 0
        self.cooldown = 0
        self.ready = False
        self.keys = {"left": False, "right": False, "up": False, "down": False}

    def move(self):
        dx, dy = 0.0, 0.0
        if self.keys["left"]: dx -= 1.0
        if self.keys["right"]: dx += 1.0
        if self.keys["up"]: dy -= 1.0
        if self.keys["down"]: dy += 1.0

        self.x = max(0.0, min(MAP_WIDTH - 1.0, self.x + dx * PLAYER_SPEED))
        self.y = max(self.y_min, min(self.y_max, self.y + dy * PLAYER_SPEED))

    def tick_cooldown(self):
        if self.cooldown > 0:
            self.cooldown -= 1

class Bullet:
    def __init__(self, x: float, y: float, vy: float, owner: str):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = vy
        self.owner = owner

    def update(self) -> bool:
        self.y += self.vy
        return 0 <= self.y < MAP_HEIGHT

class GameLobby:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Player] = {}
        self.bullets = []
        self.running = False
        self.loop_task: Optional[asyncio.Task] = None

    def is_locked_out(self, ip: str) -> bool:
        rec = failed_attempts.get(ip)
        if rec and time.time() < rec["locked_until"]:
            return True
        return False

    def register_failure(self, ip: str) -> int:
        now = time.time()
        rec = failed_attempts.setdefault(ip, {"count": 0, "locked_until": 0.0})
        if now > rec["locked_until"]:
            rec["count"] += 1
            if rec["count"] >= MAX_FAILED_ATTEMPTS:
                rec["locked_until"] = now + LOCKOUT_DURATION
                rec["count"] = 0
        return max(0, MAX_FAILED_ATTEMPTS - rec["count"])

    def clear_failures(self, ip: str):
        if ip in failed_attempts:
            del failed_attempts[ip]

    def reset_arena(self):
        self.bullets.clear()
        players = list(self.active_connections.values())
        if len(players) >= 1:
            players[0].x, players[0].y = 25.0, 20.0
            players[0].cooldown = 0
        if len(players) >= 2:
            players[1].x, players[1].y = 25.0, 280.0
            players[1].cooldown = 0

    async def broadcast(self, data: dict):
        dead_sockets = []
        for ws in self.active_connections.keys():
            try:
                await ws.send_json(data)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            await self.disconnect(ws)

    async def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            p = self.active_connections.pop(ws)
            print(f"[Lobby] {p.username} disconnected. Active players: {len(self.active_connections)}")
            self.running = False
            self.reset_arena()
            await self.broadcast({
                "type": "player_left",
                "message": f"{p.username} left the match. Waiting for an opponent..."
            })

    async def game_tick(self):
        while len(self.active_connections) == 2 and self.running:
            p_list = list(self.active_connections.values())
            pA, pB = p_list[0], p_list[1]

            # 1. Update cooldowns & positions
            pA.tick_cooldown()
            pB.tick_cooldown()
            pA.move()
            pB.move()

            # 2. Auto-fire straight on cooldown
            if pA.cooldown == 0:
                self.bullets.append(Bullet(pA.x, pA.y, BULLET_SPEED, "A"))
                pA.cooldown = FIRE_COOLDOWN
            if pB.cooldown == 0:
                self.bullets.append(Bullet(pB.x, pB.y, -BULLET_SPEED, "B"))
                pB.cooldown = FIRE_COOLDOWN

            # 3. Update bullets & check hits
            surviving = []
            for b in self.bullets:
                if not b.update():
                    continue

                if b.owner == "B" and math.hypot(b.x - pA.x, b.y - pA.y) < HITBOX_RADIUS:
                    pB.score += 1
                    continue
                if b.owner == "A" and math.hypot(b.x - pB.x, b.y - pB.y) < HITBOX_RADIUS:
                    pA.score += 1
                    continue

                surviving.append(b)

            self.bullets = surviving

            # 4. Broadcast state frame
            state = {
                "type": "state",
                "player_a": {"x": pA.x, "y": pA.y, "score": pA.score, "name": pA.username},
                "player_b": {"x": pB.x, "y": pB.y, "score": pB.score, "name": pB.username},
                "bullets": [{"x": b.x, "y": b.y, "owner": b.owner} for b in self.bullets]
            }
            await self.broadcast(state)
            await asyncio.sleep(1.0 / TPS)

lobby = GameLobby()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_ip = ws.client.host if ws.client else "unknown"

    # Check IP Lockout
    if lobby.is_locked_out(client_ip):
        await ws.send_json({"type": "error", "message": "Too many failed attempts. Locked out for 15 minutes."})
        await ws.close(code=4003)
        return

    # Check 2-Player Capacity
    if len(lobby.active_connections) >= 2:
        await ws.send_json({"type": "error", "message": "Lobby is full (Maximum 2 players allowed). Try again later."})
        await ws.close(code=4001)
        return

    try:
        # Step 1: Wait for Auth Packet from Client
        init_data = await ws.receive_json()
        username = str(init_data.get("username", "")).strip()[:14] or "Player"
        password = str(init_data.get("password", ""))

        if password != ROOM_PASSWORD:
            remaining = lobby.register_failure(client_ip)
            msg = f"Incorrect password. {remaining} attempt(s) remaining." if remaining > 0 else "Locked out for 15 minutes."
            await ws.send_json({"type": "error", "message": msg})
            await ws.close(code=4002)
            return

        lobby.clear_failures(client_ip)

        # Assign Slot A (Top) or Slot B (Bottom)
        assigned_id = "A" if len(lobby.active_connections) == 0 else "B"
        if assigned_id == "A":
            player = Player("A", username, 25.0, 20.0, 0.0, 99.0)
        else:
            player = Player("B", username, 25.0, 280.0, 200.0, 299.0)

        lobby.active_connections[ws] = player
        print(f"[Lobby] Assigned {username} to Slot {assigned_id}. Total: {len(lobby.active_connections)}")

        await ws.send_json({
            "type": "init",
            "slot": assigned_id,
            "username": username,
            "message": f"Connected as Player {assigned_id} ({username})"
        })

        # Step 2: Handle Incoming Client Events
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "ready":
                player.ready = True
                p_list = list(lobby.active_connections.values())
                if len(p_list) == 2 and all(p.ready for p in p_list):
                    lobby.running = True
                    lobby.reset_arena()
                    await lobby.broadcast({"type": "start"})
                    if lobby.loop_task is None or lobby.loop_task.done():
                        lobby.loop_task = asyncio.create_task(lobby.game_tick())
                else:
                    await lobby.broadcast({"type": "waiting_ready", "player": player.username})

            elif mtype == "keys":
                player.keys = {
                    "left": bool(msg.get("left", False)),
                    "right": bool(msg.get("right", False)),
                    "up": bool(msg.get("up", False)),
                    "down": bool(msg.get("down", False))
                }

    except WebSocketDisconnect:
        await lobby.disconnect(ws)
    except Exception as e:
        print(f"[Error] {e}")
        await lobby.disconnect(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
